#ifndef CTYPEDEFS_ALLOC_ARENA_H
#define CTYPEDEFS_ALLOC_ARENA_H

/* Bump allocator over mmap'd blocks: advances an offset, no per-object
   free. Growth adds new blocks instead of reallocating; earlier pointers
   stay valid until the arena resets or is destroyed. */

#include "../types.h"

typedef struct ArenaBlock ArenaBlock;

typedef struct Arena {
    ArenaBlock *current;
    ArenaBlock *base_block;
    usize default_block_size;
} Arena;

typedef struct ArenaMark {
    ArenaBlock *block;
    usize offset;
} ArenaMark;

/* initial_capacity is a hint, rounded up to a page; 0 picks a default. */
Arena arena_create(usize initial_capacity);
void arena_destroy(Arena *a);

void *arena_alloc(Arena *a, usize size);
/* align must be a power of two, max 64 (block headers only guarantee that much). */
void *arena_alloc_aligned(Arena *a, usize size, usize align);

ArenaMark arena_mark(const Arena *a);
void arena_reset_to(Arena *a, ArenaMark mark);
void arena_reset(Arena *a);

usize arena_bytes_used(const Arena *a);
usize arena_bytes_reserved(const Arena *a);

#endif /* CTYPEDEFS_ALLOC_ARENA_H */
