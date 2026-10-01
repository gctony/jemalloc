#include "jemalloc/internal/jemalloc_preamble.h"

#include "jemalloc/internal/hpa.h"
#include "jemalloc/internal/hpa_central.h"
#include "jemalloc/internal/witness.h"

typedef union hpa_chunk_slot_u hpa_chunk_slot_t;
union hpa_chunk_slot_u {
	hpdata_t      hpdata;
	unsigned char padding[CACHELINE_CEILING(sizeof(hpdata_t))];
};

struct hpa_chunk_s {
	/* Keep slots first so that a pageslab and its index identify the chunk. */
	hpa_chunk_slot_t slots[HPA_CHUNK_NSLOTS];
	void            *addr;
	fb_group_t       allocated[FB_NGROUPS(HPA_CHUNK_NSLOTS)];
	size_t           nallocated;
	size_t           hw;
	ql_elm(hpa_chunk_t) link;
	ql_elm(hpa_chunk_t) free_link;
};

static hpa_chunk_t *
hpa_central_desc_pop(hpa_central_t *central) {
	hpa_chunk_t *chunk = ql_first(&central->free_descs);
	if (chunk != NULL) {
		ql_remove(&central->free_descs, chunk, free_link);
	}
	return chunk;
}

static void
hpa_central_desc_push(hpa_central_t *central, hpa_chunk_t *chunk) {
	ql_elm_new(chunk, free_link);
	ql_head_insert(&central->free_descs, chunk, free_link);
}

static void
hpa_central_chunk_insert(hpa_central_t *central, hpa_chunk_t *chunk) {
	ql_elm_new(chunk, link);
	hpa_chunk_t *iter;
	ql_foreach (iter, &central->chunks, link) {
		if ((uintptr_t)chunk->addr < (uintptr_t)iter->addr) {
			ql_before_insert(&central->chunks, iter, chunk, link);
			return;
		}
	}
	ql_tail_insert(&central->chunks, chunk, link);
}

static void *
hpa_central_map(hpa_central_t *central) {
	void *addr = central->hooks.map(HPA_CHUNK_SIZE);
	if (addr == NULL) {
		return NULL;
	}
	assert(HUGEPAGE_ADDR2BASE(addr) == addr);
	if (ALIGNMENT_ADDR2BASE(addr, HPA_CHUNK_SIZE) == addr) {
		return addr;
	}

	central->hooks.unmap(addr, HPA_CHUNK_SIZE);

	const size_t alloc_size = 2 * HPA_CHUNK_SIZE - HUGEPAGE;
	void        *alloc = central->hooks.map(alloc_size);
	if (alloc == NULL) {
		return NULL;
	}
	assert(HUGEPAGE_ADDR2BASE(alloc) == alloc);

	addr = (void *)ALIGNMENT_CEILING((uintptr_t)alloc, HPA_CHUNK_SIZE);
	size_t lead = (uintptr_t)addr - (uintptr_t)alloc;
	size_t trail = alloc_size - lead - HPA_CHUNK_SIZE;
	if (lead != 0) {
		central->hooks.unmap(alloc, lead);
	}
	if (trail != 0) {
		central->hooks.unmap(
		    (void *)((byte_t *)addr + HPA_CHUNK_SIZE), trail);
	}
	return addr;
}

bool
hpa_central_init(
    hpa_central_t *central, base_t *base, const hpa_hooks_t *hooks) {
	/* malloc_conf processing should have filtered out these cases. */
	assert(hpa_supported());
	if (malloc_mutex_init(&central->grow_mtx, "hpa_central_grow",
	        WITNESS_RANK_HPA_CENTRAL_GROW, malloc_mutex_rank_exclusive)) {
		return true;
	}
	if (malloc_mutex_init(&central->mtx, "hpa_central",
	        WITNESS_RANK_HPA_CENTRAL, malloc_mutex_rank_exclusive)) {
		return true;
	}

	ql_new(&central->chunks);
	central->spare = NULL;
	ql_new(&central->free_descs);
	memset(&central->stats, 0, sizeof(central->stats));
	central->base = base;
	central->hooks = *hooks;
	return false;
}

static hpdata_t *
hpa_central_extract_locked(
    hpa_central_t *central, bool *fresh, hpa_chunk_t **r_chunk) {
	hpa_chunk_t *chunk;
	ql_foreach (chunk, &central->chunks, link) {
		if (chunk->nallocated == HPA_CHUNK_NSLOTS) {
			continue;
		}
		size_t ind = fb_ffu(chunk->allocated, HPA_CHUNK_NSLOTS, 0);
		assert(ind < HPA_CHUNK_NSLOTS);
		assert(ind <= chunk->hw);
		*fresh = ind == chunk->hw;
		if (*fresh) {
			chunk->hw++;
		}
		fb_set(chunk->allocated, HPA_CHUNK_NSLOTS, ind);
		chunk->nallocated++;
		if (central->spare == chunk) {
			central->spare = NULL;
			central->stats.nspare = 0;
		}
		central->stats.nactive++;
		central->stats.nfree--;
		central->stats.nextracts++;
		if (!*fresh) {
			central->stats.nreuses++;
		}
		*r_chunk = chunk;
		return &chunk->slots[ind].hpdata;
	}
	return NULL;
}

static void
hpa_central_slot_init(
    hpdata_t *ps, void *addr, uint64_t age, bool start_as_huge, bool fresh) {
	bool purged_when_empty_and_huge = false;
	if (!fresh) {
		assert(hpdata_empty(ps));
		assert(hpdata_ntouched_get(ps) == 0);
		assert(!hpdata_in_psset_get(ps));
		purged_when_empty_and_huge =
		    hpdata_purged_when_empty_and_huge_get(ps);
	}
	hpdata_init(ps, addr, age, start_as_huge);
	if (!fresh) {
		hpdata_purged_when_empty_and_huge_set(
		    ps, purged_when_empty_and_huge);
	}
}

hpdata_t *
hpa_central_extract(tsdn_t *tsdn, hpa_central_t *central, size_t size,
    uint64_t age, bool hugify_eager, bool *oom) {
	/* Don't yet support big allocations; these should get filtered out. */
	assert(size <= HUGEPAGE);
	/* The local shard must have serialized attempts to grow. */
	witness_assert_positive_depth_to_rank(
	    tsdn_witness_tsdp_get(tsdn), WITNESS_RANK_HPA_SHARD_GROW);
	*oom = false;

	bool         fresh;
	hpa_chunk_t *chunk;
	malloc_mutex_lock(tsdn, &central->mtx);
	hpdata_t *ps = hpa_central_extract_locked(central, &fresh, &chunk);
	malloc_mutex_unlock(tsdn, &central->mtx);
	if (ps != NULL) {
		bool start_as_huge = hugify_eager
		    || (init_system_thp_mode == system_thp_mode_always
		        && opt_experimental_hpa_start_huge_if_thp_always);
		size_t ind = ((byte_t *)ps - (byte_t *)chunk->slots)
		    / sizeof(hpa_chunk_slot_t);
		void *addr = (void *)((byte_t *)chunk->addr + ind * HUGEPAGE);
		hpa_central_slot_init(ps, addr, age, start_as_huge, fresh);
		return ps;
	}

	malloc_mutex_lock(tsdn, &central->grow_mtx);
	malloc_mutex_lock(tsdn, &central->mtx);
	ps = hpa_central_extract_locked(central, &fresh, &chunk);
	if (ps != NULL) {
		malloc_mutex_unlock(tsdn, &central->mtx);
		malloc_mutex_unlock(tsdn, &central->grow_mtx);
		bool start_as_huge = hugify_eager
		    || (init_system_thp_mode == system_thp_mode_always
		        && opt_experimental_hpa_start_huge_if_thp_always);
		size_t ind = ((byte_t *)ps - (byte_t *)chunk->slots)
		    / sizeof(hpa_chunk_slot_t);
		void *addr = (void *)((byte_t *)chunk->addr + ind * HUGEPAGE);
		hpa_central_slot_init(ps, addr, age, start_as_huge, fresh);
		return ps;
	}
	assert(central->spare == NULL);
	chunk = hpa_central_desc_pop(central);
	malloc_mutex_unlock(tsdn, &central->mtx);

	if (chunk == NULL) {
		chunk = (hpa_chunk_t *)base_alloc(
		    tsdn, central->base, sizeof(hpa_chunk_t), CACHELINE);
		if (chunk == NULL) {
			*oom = true;
			malloc_mutex_unlock(tsdn, &central->grow_mtx);
			return NULL;
		}
	}

	void *addr = hpa_central_map(central);
	if (addr == NULL) {
		malloc_mutex_lock(tsdn, &central->mtx);
		hpa_central_desc_push(central, chunk);
		malloc_mutex_unlock(tsdn, &central->mtx);
		*oom = true;
		malloc_mutex_unlock(tsdn, &central->grow_mtx);
		return NULL;
	}
	if (hugify_eager) {
		/* Hugification style is global, so this is needed once per chunk. */
		central->hooks.hugify(addr, HPA_CHUNK_SIZE, /* sync */ false);
	}

	chunk->addr = addr;
	fb_init(chunk->allocated, HPA_CHUNK_NSLOTS);
	chunk->nallocated = 0;
	chunk->hw = 0;
	malloc_mutex_lock(tsdn, &central->mtx);
	hpa_central_chunk_insert(central, chunk);
	central->stats.nchunks++;
	central->stats.nchunk_maps++;
	central->stats.nfree += HPA_CHUNK_NSLOTS;
	ps = hpa_central_extract_locked(central, &fresh, &chunk);
	assert(ps != NULL);
	assert(fresh);
	malloc_mutex_unlock(tsdn, &central->mtx);
	malloc_mutex_unlock(tsdn, &central->grow_mtx);

	bool start_as_huge = hugify_eager
	    || (init_system_thp_mode == system_thp_mode_always
	        && opt_experimental_hpa_start_huge_if_thp_always);
	hpa_central_slot_init(ps, addr, age, start_as_huge, fresh);
	return ps;
}

void
hpa_central_dalloc(tsdn_t *tsdn, hpa_central_t *central, hpdata_t *ps) {
	witness_assert_depth_to_rank(
	    tsdn_witness_tsdp_get(tsdn), WITNESS_RANK_HPA_SHARD, 0);
	assert(hpdata_empty(ps));
	assert(!hpdata_in_psset_get(ps));
	assert(!hpdata_changing_state_get(ps));

	bool purged = hpdata_ntouched_get(ps) > 0;
	if (purged) {
		hpdata_purged_when_empty_and_huge_set(ps, hpdata_huge_get(ps));
		central->hooks.purge(hpdata_addr_get(ps), HUGEPAGE);
		hpdata_alloc_allowed_set(ps, false);
		hpdata_purge_state_t purge_state;
		size_t               nranges;
		hpdata_purge_begin(ps, &purge_state, &nranges);
		void  *purge_addr;
		size_t purge_size;
		while (hpdata_purge_next(
		    ps, &purge_state, &purge_addr, &purge_size)) {
		}
		hpdata_purge_end(ps, &purge_state);
		hpdata_alloc_allowed_set(ps, true);
		if (hpdata_huge_get(ps)) {
			hpdata_dehugify(ps);
		}
	}

	void  *addr = hpdata_addr_get(ps);
	void  *chunk_addr = ALIGNMENT_ADDR2BASE(addr, HPA_CHUNK_SIZE);
	size_t ind = ((uintptr_t)addr - (uintptr_t)chunk_addr) >> LG_HUGEPAGE;
	assert(ind < HPA_CHUNK_NSLOTS);
	hpa_chunk_t *chunk = (hpa_chunk_t *)((byte_t *)ps
	    - ind * sizeof(hpa_chunk_slot_t));
	assert(chunk->addr == chunk_addr);
	assert(&chunk->slots[ind].hpdata == ps);

	void *victim_addr = NULL;
	malloc_mutex_lock(tsdn, &central->mtx);
	assert(fb_get(chunk->allocated, HPA_CHUNK_NSLOTS, ind));
	fb_unset(chunk->allocated, HPA_CHUNK_NSLOTS, ind);
	assert(chunk->nallocated > 0);
	chunk->nallocated--;
	central->stats.nactive--;
	central->stats.nfree++;
	central->stats.ndallocs++;
	if (purged) {
		central->stats.ndalloc_purges++;
	}
	if (chunk->nallocated == 0) {
		assert(fb_empty(chunk->allocated, HPA_CHUNK_NSLOTS));
		if (central->spare == NULL) {
			central->spare = chunk;
			central->stats.nspare = 1;
		} else {
			hpa_chunk_t *victim;
			if ((uintptr_t)central->spare->addr
			    < (uintptr_t)chunk->addr) {
				victim = chunk;
			} else {
				victim = central->spare;
				central->spare = chunk;
			}
			victim_addr = victim->addr;
			ql_remove(&central->chunks, victim, link);
			hpa_central_desc_push(central, victim);
			central->stats.nchunks--;
			central->stats.nfree -= HPA_CHUNK_NSLOTS;
			central->stats.nchunk_unmaps++;
		}
	}
	malloc_mutex_unlock(tsdn, &central->mtx);

	if (victim_addr != NULL) {
		central->hooks.unmap(victim_addr, HPA_CHUNK_SIZE);
	}
}

void
hpa_central_stats_read(
    tsdn_t *tsdn, hpa_central_t *central, hpa_central_stats_t *stats) {
	malloc_mutex_lock(tsdn, &central->mtx);
	*stats = central->stats;
	malloc_mutex_unlock(tsdn, &central->mtx);
}

void
hpa_central_prefork(tsdn_t *tsdn, hpa_central_t *central) {
	malloc_mutex_prefork(tsdn, &central->grow_mtx);
	malloc_mutex_prefork(tsdn, &central->mtx);
}

void
hpa_central_postfork_parent(tsdn_t *tsdn, hpa_central_t *central) {
	malloc_mutex_postfork_parent(tsdn, &central->mtx);
	malloc_mutex_postfork_parent(tsdn, &central->grow_mtx);
}

void
hpa_central_postfork_child(tsdn_t *tsdn, hpa_central_t *central) {
	malloc_mutex_postfork_child(tsdn, &central->mtx);
	malloc_mutex_postfork_child(tsdn, &central->grow_mtx);
}
