#include "test/jemalloc_test.h"

#include "test/hpa.h"

#include "jemalloc/internal/hpa.h"
#include "jemalloc/internal/hpa_central.h"

#define HPA_TEST_ARENA_IND 125
#define HPA_TEST_MAX_CALLS 16

static bool     hpa_test_map_fail;
static unsigned hpa_test_map_calls;
static unsigned hpa_test_unmap_calls;
static unsigned hpa_test_purge_calls;
static unsigned hpa_test_hugify_calls;
static void    *hpa_test_map_results[HPA_TEST_MAX_CALLS];
static void    *hpa_test_map_addrs[HPA_TEST_MAX_CALLS];
static size_t   hpa_test_map_sizes[HPA_TEST_MAX_CALLS];
static void    *hpa_test_unmap_addrs[HPA_TEST_MAX_CALLS];
static size_t   hpa_test_unmap_sizes[HPA_TEST_MAX_CALLS];
static void    *hpa_test_last_purge;
static size_t   hpa_test_last_purge_size;
static void    *hpa_test_last_hugify;
static size_t   hpa_test_last_hugify_size;

static void
hpa_test_hooks_reset(void) {
	hpa_test_map_fail = false;
	hpa_test_map_calls = 0;
	hpa_test_unmap_calls = 0;
	hpa_test_purge_calls = 0;
	hpa_test_hugify_calls = 0;
	memset(hpa_test_map_results, 0, sizeof(hpa_test_map_results));
	memset(hpa_test_map_addrs, 0, sizeof(hpa_test_map_addrs));
	memset(hpa_test_map_sizes, 0, sizeof(hpa_test_map_sizes));
	memset(hpa_test_unmap_addrs, 0, sizeof(hpa_test_unmap_addrs));
	memset(hpa_test_unmap_sizes, 0, sizeof(hpa_test_unmap_sizes));
	hpa_test_last_purge = NULL;
	hpa_test_last_purge_size = 0;
	hpa_test_last_hugify = NULL;
	hpa_test_last_hugify_size = 0;
}

static void
hpa_test_map_result(unsigned ind, uintptr_t addr) {
	assert_u_lt(ind, HPA_TEST_MAX_CALLS, "Too many scripted map results");
	hpa_test_map_results[ind] = (void *)addr;
}

static void *
hpa_test_map(size_t size) {
	unsigned ind = hpa_test_map_calls++;
	assert_u_lt(ind, HPA_TEST_MAX_CALLS, "Too many map calls");
	hpa_test_map_sizes[ind] = size;
	if (hpa_test_map_fail) {
		return NULL;
	}
	void *ret = hpa_test_map_results[ind];
	assert_ptr_not_null(ret, "Missing scripted map result");
	hpa_test_map_addrs[ind] = ret;
	return ret;
}

static void
hpa_test_unmap(void *ptr, size_t size) {
	unsigned ind = hpa_test_unmap_calls++;
	assert_u_lt(ind, HPA_TEST_MAX_CALLS, "Too many unmap calls");
	hpa_test_unmap_addrs[ind] = ptr;
	hpa_test_unmap_sizes[ind] = size;
}

static void
hpa_test_purge(void *ptr, size_t size) {
	hpa_test_purge_calls++;
	hpa_test_last_purge = ptr;
	hpa_test_last_purge_size = size;
}

static bool
hpa_test_hugify(void *ptr, size_t size, bool sync) {
	hpa_test_hugify_calls++;
	hpa_test_last_hugify = ptr;
	hpa_test_last_hugify_size = size;
	return false;
}

static void
hpa_test_dehugify(void *ptr, size_t size) {}

static void
hpa_test_curtime(nstime_t *r_time, bool first_reading) {
	nstime_init(r_time, 0);
}

static uint64_t
hpa_test_ms_since(nstime_t *r_time) {
	return 0;
}

static bool
hpa_test_vectorized_purge(void *vec, size_t vlen, size_t nbytes) {
	return true;
}

static hpa_hooks_t hpa_test_hooks = {hpa_test_map, hpa_test_unmap,
    hpa_test_purge, hpa_test_hugify, hpa_test_dehugify, hpa_test_curtime,
    hpa_test_ms_since, hpa_test_vectorized_purge};

static bool     hpa_base_fail_after_new;
static unsigned hpa_base_alloc_calls;

static void
hpa_base_hooks_reset(void) {
	hpa_base_fail_after_new = false;
	hpa_base_alloc_calls = 0;
}

static void *
hpa_base_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
	if (hpa_base_fail_after_new && hpa_base_alloc_calls > 0) {
		hpa_base_alloc_calls++;
		return NULL;
	}
	hpa_base_alloc_calls++;
	return pages_map(new_addr, size, alignment, commit);
}

static bool
hpa_base_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
	pages_unmap(addr, size);
	return false;
}

static void
hpa_base_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
	pages_unmap(addr, size);
}

static extent_hooks_t hpa_base_hooks = {
    hpa_base_alloc, hpa_base_dalloc, hpa_base_destroy, NULL, /* commit */
    NULL,                                                    /* decommit */
    NULL,                                                    /* purge_lazy */
    NULL,                                                    /* purge_forced */
    NULL,                                                    /* split */
    NULL                                                     /* merge */
};

static hpdata_t *
hpa_central_extract_with_lock(hpa_central_t *central, malloc_mutex_t *mtx,
    uint64_t age, bool hugify_eager, bool *oom) {
	tsdn_t *tsdn = tsdn_fetch();
	malloc_mutex_lock(tsdn, mtx);
	hpdata_t *ret = hpa_central_extract(
	    tsdn, central, PAGE, age, hugify_eager, oom);
	malloc_mutex_unlock(tsdn, mtx);
	return ret;
}

static void
hpa_test_shard_grow_mtx_init(malloc_mutex_t *mtx) {
	assert_false(
	    malloc_mutex_init(mtx, "hpa_test_shard_grow",
	        WITNESS_RANK_HPA_SHARD_GROW, malloc_mutex_rank_exclusive),
	    "Unexpected mutex initialization failure");
}

static base_t *
hpa_test_central_init(hpa_central_t *central, malloc_mutex_t *grow_mtx,
    unsigned arena_ind, const extent_hooks_t *base_hooks) {
	tsdn_t *tsdn = tsdn_fetch();
	base_t *base = base_new(tsdn, arena_ind, base_hooks,
	    /* metadata_use_hooks */ true);
	assert_ptr_not_null(base, "Unexpected base_new failure");
	assert_false(hpa_central_init(central, base, &hpa_test_hooks),
	    "Unexpected hpa_central_init failure");
	hpa_test_shard_grow_mtx_init(grow_mtx);
	return base;
}

static hpa_central_stats_t
hpa_test_stats(hpa_central_t *central) {
	hpa_central_stats_t stats;
	hpa_central_stats_read(tsdn_fetch(), central, &stats);
	return stats;
}

TEST_BEGIN(test_hpa_central_mapping) {
	test_skip_if(!hpa_supported() || hpa_hugepage_size_exceeds_limit());
	hpa_test_hooks_reset();

	uintptr_t aligned = 4 * (uintptr_t)HPA_CHUNK_SIZE;
	hpa_test_map_result(0, aligned);
	hpa_central_t  central;
	malloc_mutex_t grow_mtx;
	base_t        *base = hpa_test_central_init(&central, &grow_mtx,
	           HPA_TEST_ARENA_IND, &ehooks_default_extent_hooks);

	bool      oom;
	hpdata_t *ps0 = hpa_central_extract_with_lock(
	    &central, &grow_mtx, 1000, /* hugify_eager */ false, &oom);
	hpdata_t *ps1 = hpa_central_extract_with_lock(
	    &central, &grow_mtx, 1001, /* hugify_eager */ false, &oom);
	expect_ptr_eq((void *)aligned, hpdata_addr_get(ps0),
	    "Aligned mapping should use slot zero");
	expect_ptr_eq((void *)(aligned + HUGEPAGE), hpdata_addr_get(ps1),
	    "Slots should ascend by one hugepage");
	expect_u_eq(1, hpa_test_map_calls, "Aligned mapping needs one map");
	expect_u_eq(0, hpa_test_unmap_calls, "Aligned mapping needs no trim");
	base_delete(tsdn_fetch(), base);

	hpa_test_hooks_reset();
	uintptr_t first = 8 * (uintptr_t)HPA_CHUNK_SIZE + HUGEPAGE;
	uintptr_t over = 12 * (uintptr_t)HPA_CHUNK_SIZE + 2 * HUGEPAGE;
	hpa_test_map_result(0, first);
	hpa_test_map_result(1, over);
	base = hpa_test_central_init(&central, &grow_mtx,
	    HPA_TEST_ARENA_IND + 1, &ehooks_default_extent_hooks);
	ps0 = hpa_central_extract_with_lock(
	    &central, &grow_mtx, 1, /* hugify_eager */ false, &oom);
	uintptr_t trimmed = 13 * (uintptr_t)HPA_CHUNK_SIZE;
	expect_ptr_eq((void *)trimmed, hpdata_addr_get(ps0),
	    "Misaligned mapping should be trimmed to a chunk boundary");
	expect_u_eq(2, hpa_test_map_calls, "Misaligned mapping should retry");
	expect_zu_eq(HPA_CHUNK_SIZE, hpa_test_map_sizes[0],
	    "Unexpected optimistic mapping size");
	expect_zu_eq(2 * HPA_CHUNK_SIZE - HUGEPAGE, hpa_test_map_sizes[1],
	    "Unexpected overmapping size");
	expect_u_eq(3, hpa_test_unmap_calls,
	    "Misaligned mapping should discard and trim both sides");
	for (unsigned i = 0; i < hpa_test_unmap_calls; i++) {
		expect_zu_gt(hpa_test_unmap_sizes[i], 0,
		    "Zero-length trims must be skipped");
	}
	base_delete(tsdn_fetch(), base);
}
TEST_END

TEST_BEGIN(test_hpa_central_hugify_and_growth) {
	test_skip_if(!hpa_supported() || hpa_hugepage_size_exceeds_limit());
	hpa_test_hooks_reset();
	uintptr_t first = 16 * (uintptr_t)HPA_CHUNK_SIZE;
	uintptr_t second = first + HPA_CHUNK_SIZE;
	hpa_test_map_result(0, first);
	hpa_test_map_result(1, second);
	hpa_central_t  central;
	malloc_mutex_t grow_mtx;
	base_t        *base = hpa_test_central_init(&central, &grow_mtx,
	           HPA_TEST_ARENA_IND + 2, &ehooks_default_extent_hooks);

	bool oom;
	for (size_t i = 0; i < HPA_CHUNK_NSLOTS; i++) {
		hpdata_t *ps = hpa_central_extract_with_lock(
		    &central, &grow_mtx, i, /* hugify_eager */ true, &oom);
		expect_ptr_eq((void *)(first + i * HUGEPAGE),
		    hpdata_addr_get(ps), "Unexpected slot address");
	}
	expect_u_eq(1, hpa_test_hugify_calls,
	    "Eager mode should hugify once per chunk");
	expect_ptr_eq((void *)first, hpa_test_last_hugify,
	    "Hugify should cover the first chunk");
	expect_zu_eq(HPA_CHUNK_SIZE, hpa_test_last_hugify_size,
	    "Hugify should cover a whole chunk");

	hpdata_t *ps = hpa_central_extract_with_lock(&central, &grow_mtx,
	    HPA_CHUNK_NSLOTS, /* hugify_eager */ false, &oom);
	expect_ptr_eq((void *)second, hpdata_addr_get(ps),
	    "An exhausted chunk should cause another mapping");
	expect_u_eq(2, hpa_test_map_calls, "Expected two chunk mappings");
	expect_u_eq(
	    1, hpa_test_hugify_calls, "A lazy chunk should not be hugified");
	hpa_central_stats_t stats = hpa_test_stats(&central);
	expect_zu_eq(2, stats.nchunks, "Unexpected mapped chunk count");
	expect_zu_eq(HPA_CHUNK_NSLOTS + 1, stats.nactive,
	    "Unexpected active slot count");
	base_delete(tsdn_fetch(), base);
}
TEST_END

TEST_BEGIN(test_hpa_central_reuse_and_spare) {
	test_skip_if(!hpa_supported() || hpa_hugepage_size_exceeds_limit());
	hpa_test_hooks_reset();
	uintptr_t first = 20 * (uintptr_t)HPA_CHUNK_SIZE;
	uintptr_t second = first + HPA_CHUNK_SIZE;
	hpa_test_map_result(0, first);
	hpa_test_map_result(1, second);
	hpa_central_t  central;
	malloc_mutex_t grow_mtx;
	base_t        *base = hpa_test_central_init(&central, &grow_mtx,
	           HPA_TEST_ARENA_IND + 3, &ehooks_default_extent_hooks);

	hpdata_t *slots[HPA_CHUNK_NSLOTS + 1];
	bool      oom;
	for (size_t i = 0; i < HPA_CHUNK_NSLOTS + 1; i++) {
		slots[i] = hpa_central_extract_with_lock(&central, &grow_mtx, i,
		    /* hugify_eager */ false, &oom);
	}
	hpdata_purged_when_empty_and_huge_set(slots[7], true);
	hpa_central_dalloc(tsdn_fetch(), &central, slots[7]);
	hpdata_t *reused = hpa_central_extract_with_lock(
	    &central, &grow_mtx, 999, /* hugify_eager */ false, &oom);
	expect_ptr_eq(
	    slots[7], reused, "First fit should reuse the returned slot");
	expect_true(hpdata_purged_when_empty_and_huge_get(reused),
	    "Reuse should preserve the purged-while-huge flag");
	expect_false(hpdata_purged_when_empty_and_huge_get(slots[8]),
	    "Fresh slots should not inherit the purged flag");

	hpa_central_dalloc(tsdn_fetch(), &central, slots[HPA_CHUNK_NSLOTS]);
	hpa_central_stats_t stats = hpa_test_stats(&central);
	expect_zu_eq(1, stats.nspare, "The empty high chunk should be spare");
	for (size_t i = 0; i < HPA_CHUNK_NSLOTS; i++) {
		hpa_central_dalloc(
		    tsdn_fetch(), &central, i == 7 ? reused : slots[i]);
	}
	stats = hpa_test_stats(&central);
	expect_zu_eq(1, stats.nchunks, "Only one spare chunk should remain");
	expect_zu_eq(1, stats.nspare, "The lower chunk should become spare");
	expect_u64_eq(1, stats.nchunk_unmaps,
	    "The higher-addressed spare should be unmapped");
	expect_ptr_eq((void *)second, hpa_test_unmap_addrs[0],
	    "The higher-addressed chunk should be unmapped");

	unsigned maps_before = hpa_test_map_calls;
	reused = hpa_central_extract_with_lock(&central, &grow_mtx, 1000,
	    /* hugify_eager */ false, &oom);
	expect_ptr_eq((void *)first, hpdata_addr_get(reused),
	    "The next extraction should reuse the lower spare");
	expect_u_eq(maps_before, hpa_test_map_calls,
	    "Reusing the spare should not map another chunk");
	stats = hpa_test_stats(&central);
	expect_u64_eq(2, stats.nreuses, "Unexpected reuse count");
	base_delete(tsdn_fetch(), base);
}
TEST_END

TEST_BEGIN(test_hpa_central_dirty_return) {
	test_skip_if(!hpa_supported() || hpa_hugepage_size_exceeds_limit());
	hpa_test_hooks_reset();
	uintptr_t addr = 24 * (uintptr_t)HPA_CHUNK_SIZE;
	hpa_test_map_result(0, addr);
	hpa_central_t  central;
	malloc_mutex_t grow_mtx;
	base_t        *base = hpa_test_central_init(&central, &grow_mtx,
	           HPA_TEST_ARENA_IND + 4, &ehooks_default_extent_hooks);

	bool      oom;
	hpdata_t *ps = hpa_central_extract_with_lock(
	    &central, &grow_mtx, 0, /* hugify_eager */ false, &oom);
	hpdata_alloc_offset_t offset;
	expect_zu_eq(1, hpdata_find_alloc_offsets(ps, PAGE, &offset, 1),
	    "Expected one allocation offset");
	void *allocation = hpdata_reserve_alloc_offset(ps, PAGE, &offset);
	hpdata_post_reserve_alloc_offsets(ps, PAGE, &offset, 1);
	expect_ptr_eq((void *)addr, allocation,
	    "The first reservation should start at the pageslab address");
	hpdata_unreserve(ps, allocation, PAGE);
	hpa_central_dalloc(tsdn_fetch(), &central, ps);
	expect_u_eq(
	    1, hpa_test_purge_calls, "Dirty return should purge exactly once");
	expect_ptr_eq((void *)addr, hpa_test_last_purge,
	    "Dirty return should purge the pageslab address");
	expect_zu_eq(HUGEPAGE, hpa_test_last_purge_size,
	    "Dirty return should purge one hugepage");
	expect_zu_eq(0, hpdata_ntouched_get(ps),
	    "Returned pageslab metadata should be clean");
	hpa_central_stats_t stats = hpa_test_stats(&central);
	expect_u64_eq(
	    1, stats.ndalloc_purges, "Dirty-return purge should be counted");
	base_delete(tsdn_fetch(), base);
}
TEST_END

TEST_BEGIN(test_hpa_central_failure_paths) {
	test_skip_if(!hpa_supported() || hpa_hugepage_size_exceeds_limit());
	hpa_test_hooks_reset();

	hpa_central_t  central;
	malloc_mutex_t grow_mtx;
	base_t        *base = hpa_test_central_init(&central, &grow_mtx,
	           HPA_TEST_ARENA_IND + 5, &ehooks_default_extent_hooks);
	hpa_test_map_fail = true;
	bool      oom = false;
	hpdata_t *ps = hpa_central_extract_with_lock(&central, &grow_mtx, 1,
	    /* hugify_eager */ false, &oom);
	expect_ptr_null(ps, "Map failure should not return hpdata");
	expect_true(oom, "Map failure should report OOM");
	expect_u_eq(1, hpa_test_map_calls, "Expected one map attempt");

	hpa_test_map_fail = false;
	hpa_test_map_result(1, 28 * (uintptr_t)HPA_CHUNK_SIZE);
	ps = hpa_central_extract_with_lock(&central, &grow_mtx, 2,
	    /* hugify_eager */ false, &oom);
	expect_ptr_not_null(ps, "Retry should reuse the descriptor");
	expect_false(oom, "Successful retry should clear OOM");
	base_delete(tsdn_fetch(), base);

	hpa_base_hooks_reset();
	hpa_test_hooks_reset();
	base = base_new(tsdn_fetch(), HPA_TEST_ARENA_IND + 6, &hpa_base_hooks,
	    /* metadata_use_hooks */ true);
	assert_ptr_not_null(base, "Unexpected base_new failure");
	hpa_base_fail_after_new = true;
	while (base_alloc(tsdn_fetch(), base, sizeof(hpdata_t), CACHELINE)
	    != NULL) {
	}
	assert_false(hpa_central_init(&central, base, &hpa_test_hooks),
	    "Unexpected hpa_central_init failure");
	hpa_test_shard_grow_mtx_init(&grow_mtx);
	oom = false;
	ps = hpa_central_extract_with_lock(&central, &grow_mtx, 3,
	    /* hugify_eager */ false, &oom);
	expect_ptr_null(ps, "Metadata OOM should not return hpdata");
	expect_true(oom, "Metadata allocation failure should report OOM");
	expect_u_eq(0, hpa_test_map_calls,
	    "Metadata OOM should happen before any mapping");
	base_delete(tsdn_fetch(), base);
}
TEST_END

int
main(void) {
	return test_no_reentrancy(test_hpa_central_mapping,
	    test_hpa_central_hugify_and_growth,
	    test_hpa_central_reuse_and_spare, test_hpa_central_dirty_return,
	    test_hpa_central_failure_paths);
}
