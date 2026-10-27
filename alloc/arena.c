#include "arena.h"

#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define ARENA_DEFAULT_BLOCK_SIZE ((usize)(1u << 20)) /* 1 MiB */
#define ARENA_BLOCK_HEADER_ALIGN ((usize)64) /* max alignment */

struct ArenaBlock {
    ArenaBlock *prev;
    u8 *base;
    usize capacity;
    usize offset;
    usize mapped_size; /* base+mapped_size for munmap */
};

static usize arena_page_size(void)
{
    long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (usize)ps : 4096u;
}

static usize round_up(usize n, usize align)
{
    return (n + (align - 1)) & ~(align - 1);
}

static ArenaBlock *arena_block_new(usize usable_size)
{
    usize page = arena_page_size();
    usize header = round_up(sizeof(ArenaBlock), ARENA_BLOCK_HEADER_ALIGN);
    usize mapped = round_up(header + usable_size, page);

    void *mem = mmap(NULL, mapped, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return NULL;
    }

    ArenaBlock *block = (ArenaBlock *)mem;
    block->prev = NULL;
    block->base = (u8 *)mem + header;
    block->capacity = mapped - header;
    block->offset = 0;
    block->mapped_size = mapped;
    return block;
}

Arena arena_create(usize initial_capacity)
{
    Arena a;
    a.default_block_size = initial_capacity ? initial_capacity : ARENA_DEFAULT_BLOCK_SIZE;
    a.current = arena_block_new(a.default_block_size);
    a.base_block = a.current;
    return a;
}

void arena_destroy(Arena *a)
{
    ArenaBlock *block = a->current;
    while (block) {
        ArenaBlock *prev = block->prev;
        munmap(block, block->mapped_size);
        block = prev;
    }
    a->current = NULL;
    a->base_block = NULL;
}

static usize aligned_offset_for(const ArenaBlock *block, usize align)
{
    uintptr_t addr = (uintptr_t)block->base + (uintptr_t)block->offset;
    uintptr_t aligned_addr = round_up((usize)addr, align);
    return (usize)(aligned_addr - (uintptr_t)block->base);
}

void *arena_alloc_aligned(Arena *a, usize size, usize align)
{
    if (align == 0) {
        align = sizeof(void *);
    }

    ArenaBlock *block = a->current;
    usize aligned_offset = block ? aligned_offset_for(block, align) : 0;

    if (!block || aligned_offset + size > block->capacity) {
        usize needed = size + align; /* worst-case slack for alignment in fresh block */
        usize new_block_size = a->default_block_size;
        if (new_block_size < needed) {
            new_block_size = needed;
        }
        /* geometric growth avoids thrashing mmap on repeated overflow */
        if (block && new_block_size < block->capacity * 2) {
            new_block_size = block->capacity * 2;
        }

        ArenaBlock *fresh = arena_block_new(new_block_size);
        if (!fresh) {
            return NULL;
        }
        fresh->prev = block;
        a->current = fresh;
        block = fresh;
        aligned_offset = aligned_offset_for(block, align);
    }

    void *ptr = block->base + aligned_offset;
    block->offset = aligned_offset + size;
    return ptr;
}

void *arena_alloc(Arena *a, usize size)
{
    return arena_alloc_aligned(a, size, sizeof(void *) * 2);
}

ArenaMark arena_mark(const Arena *a)
{
    ArenaMark m;
    m.block = a->current;
    m.offset = a->current ? a->current->offset : 0;
    return m;
}

void arena_reset_to(Arena *a, ArenaMark mark)
{
    ArenaBlock *block = a->current;
    while (block && block != mark.block) {
        ArenaBlock *prev = block->prev;
        munmap(block, block->mapped_size);
        block = prev;
    }
    a->current = block;
    if (block) {
        block->offset = mark.offset;
    }
}

void arena_reset(Arena *a)
{
    ArenaMark m;
    m.block = a->base_block;
    m.offset = 0;
    arena_reset_to(a, m);
}

usize arena_bytes_used(const Arena *a)
{
    usize total = 0;
    for (ArenaBlock *b = a->current; b; b = b->prev) {
        total += b->offset;
    }
    return total;
}

usize arena_bytes_reserved(const Arena *a)
{
    usize total = 0;
    for (ArenaBlock *b = a->current; b; b = b->prev) {
        total += b->capacity;
    }
    return total;
}
