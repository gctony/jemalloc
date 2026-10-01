#ifndef JEMALLOC_INTERNAL_HPA_CENTRAL_H
#define JEMALLOC_INTERNAL_HPA_CENTRAL_H

#include "jemalloc/internal/jemalloc_preamble.h"
#include "jemalloc/internal/base.h"
#include "jemalloc/internal/fb.h"
#include "jemalloc/internal/hpa_hooks.h"
#include "jemalloc/internal/hpdata.h"
#include "jemalloc/internal/mutex.h"
#include "jemalloc/internal/ql.h"
#include "jemalloc/internal/tsd_types.h"

#if LG_HUGEPAGE > 30
#	define LG_HPA_CHUNK LG_HUGEPAGE
#else
#	define LG_HPA_CHUNK 30
#endif

#if LG_HPA_CHUNK >= (1 << (LG_SIZEOF_PTR + 3))
#	error "HPA chunk size does not fit in a pointer"
#endif

#define HPA_CHUNK_SIZE ((size_t)1 << LG_HPA_CHUNK)
#define HPA_CHUNK_NSLOTS (ZU(1) << (LG_HPA_CHUNK - LG_HUGEPAGE))

typedef struct hpa_chunk_s hpa_chunk_t;

typedef struct hpa_central_stats_s hpa_central_stats_t;
struct hpa_central_stats_s {
	/* Gauges, in chunks or hugepages. */
	size_t nchunks;
	size_t nspare;
	size_t nactive;
	size_t nfree;
	/* Monotonic event counters. */
	uint64_t nchunk_maps;
	uint64_t nchunk_unmaps;
	uint64_t nextracts;
	uint64_t nreuses;
	uint64_t ndallocs;
	uint64_t ndalloc_purges;
};

typedef struct hpa_central_s hpa_central_t;
struct hpa_central_s {
	/* Serializes mappings, without blocking extracts from existing chunks. */
	malloc_mutex_t grow_mtx;
	/* Guards chunks, descriptors, the spare chunk, and stats. */
	malloc_mutex_t mtx;
	ql_head(hpa_chunk_t) chunks;
	hpa_chunk_t *spare;
	ql_head(hpa_chunk_t) free_descs;
	hpa_central_stats_t stats;

	/* Source for metadata. */
	base_t *base;

	/* The HPA hooks. */
	hpa_hooks_t hooks;
};

bool hpa_central_init(
    hpa_central_t *central, base_t *base, const hpa_hooks_t *hooks);

hpdata_t *hpa_central_extract(tsdn_t *tsdn, hpa_central_t *central, size_t size,
    uint64_t age, bool hugify_eager, bool *oom);
void hpa_central_dalloc(tsdn_t *tsdn, hpa_central_t *central, hpdata_t *ps);

void hpa_central_stats_read(
    tsdn_t *tsdn, hpa_central_t *central, hpa_central_stats_t *stats);

void hpa_central_prefork(tsdn_t *tsdn, hpa_central_t *central);
void hpa_central_postfork_parent(tsdn_t *tsdn, hpa_central_t *central);
void hpa_central_postfork_child(tsdn_t *tsdn, hpa_central_t *central);

#endif /* JEMALLOC_INTERNAL_HPA_CENTRAL_H */
