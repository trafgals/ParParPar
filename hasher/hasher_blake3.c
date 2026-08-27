/*
 * BLAKE3 portable C implementation, ported from BLAKE3-team/BLAKE3 v1.8.7.
 * See hasher_blake3.h for licensing and scope.
 *
 * This file contains: the IV + MSG_SCHEDULE + helper utilities (impl.h bits),
 * the main hasher state machine (blake3.c bits), and the portable compress
 * function (blake3_portable.c bits). All merged into one TU so the hasher/
 * library only needs to add a single .c file to its build.
 */
#include "hasher_blake3.h"

#include <string.h>

/* ---- impl.h bits ---- */

#if defined(__GNUC__) || defined(__clang__)
#define INLINE static inline __attribute__((always_inline))
#else
#define INLINE static inline
#endif

enum blake3_flags {
    CHUNK_START         = 1 << 0,
    CHUNK_END           = 1 << 1,
    PARENT              = 1 << 2,
    ROOT                = 1 << 3,
    KEYED_HASH          = 1 << 4,
    DERIVE_KEY_CONTEXT  = 1 << 5,
    DERIVE_KEY_MATERIAL = 1 << 6,
};

#define MAX_SIMD_DEGREE 1
#define MAX_SIMD_DEGREE_OR_2 2

static const uint32_t IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static const uint8_t MSG_SCHEDULE[7][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};

INLINE uint32_t rotr32(uint32_t w, uint32_t c) {
    return (w >> c) | (w << (32 - c));
}

INLINE unsigned int popcnt(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned int)__builtin_popcountll(x);
#else
    unsigned int count = 0;
    while (x != 0) { count += 1; x &= x - 1; }
    return count;
#endif
}

INLINE uint64_t round_down_to_power_of_2(uint64_t x) {
    /* x is always > 0 here in callers; special-case 0 -> 1 for safety. */
    if (x == 0) return 1;
    unsigned int r = 0;
    while ((x >> 1) != 0) { x >>= 1; r++; }
    return 1ULL << r;
}

INLINE uint32_t counter_low(uint64_t counter)  { return (uint32_t)counter; }
INLINE uint32_t counter_high(uint64_t counter) { return (uint32_t)(counter >> 32); }

INLINE uint32_t load32(const void *src) {
    const uint8_t *p = (const uint8_t *)src;
    return ((uint32_t)p[0])
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

INLINE void store32(void *dst, uint32_t w) {
    uint8_t *p = (uint8_t *)dst;
    p[0] = (uint8_t)(w);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
}

INLINE void store_cv_words(uint8_t bytes_out[32], const uint32_t cv_words[8]) {
    store32(&bytes_out[ 0], cv_words[0]);
    store32(&bytes_out[ 4], cv_words[1]);
    store32(&bytes_out[ 8], cv_words[2]);
    store32(&bytes_out[12], cv_words[3]);
    store32(&bytes_out[16], cv_words[4]);
    store32(&bytes_out[20], cv_words[5]);
    store32(&bytes_out[24], cv_words[6]);
    store32(&bytes_out[28], cv_words[7]);
}

/* ---- portable compress ---- */

INLINE void g_fn(uint32_t *state, size_t a, size_t b, size_t c, size_t d,
                 uint32_t x, uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

INLINE void round_fn(uint32_t state[16], const uint32_t *msg, size_t round) {
    const uint8_t *schedule = MSG_SCHEDULE[round];
    /* Mix the columns. */
    g_fn(state, 0, 4, 8, 12, msg[schedule[ 0]], msg[schedule[ 1]]);
    g_fn(state, 1, 5, 9, 13, msg[schedule[ 2]], msg[schedule[ 3]]);
    g_fn(state, 2, 6, 10, 14, msg[schedule[ 4]], msg[schedule[ 5]]);
    g_fn(state, 3, 7, 11, 15, msg[schedule[ 6]], msg[schedule[ 7]]);
    /* Mix the rows. */
    g_fn(state, 0, 5, 10, 15, msg[schedule[ 8]], msg[schedule[ 9]]);
    g_fn(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g_fn(state, 2, 7,  8, 13, msg[schedule[12]], msg[schedule[13]]);
    g_fn(state, 3, 4,  9, 14, msg[schedule[14]], msg[schedule[15]]);
}

INLINE void compress_pre(uint32_t state[16], const uint32_t cv[8],
                         const uint8_t block[BLAKE3_BLOCK_LEN],
                         uint8_t block_len, uint64_t counter, uint8_t flags) {
    uint32_t block_words[16];
    for (size_t i = 0; i < 16; i++) {
        block_words[i] = load32(block + 4 * i);
    }
    state[ 0] = cv[0];
    state[ 1] = cv[1];
    state[ 2] = cv[2];
    state[ 3] = cv[3];
    state[ 4] = cv[4];
    state[ 5] = cv[5];
    state[ 6] = cv[6];
    state[ 7] = cv[7];
    state[ 8] = IV[0];
    state[ 9] = IV[1];
    state[10] = IV[2];
    state[11] = IV[3];
    state[12] = counter_low(counter);
    state[13] = counter_high(counter);
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    round_fn(state, block_words, 0);
    round_fn(state, block_words, 1);
    round_fn(state, block_words, 2);
    round_fn(state, block_words, 3);
    round_fn(state, block_words, 4);
    round_fn(state, block_words, 5);
    round_fn(state, block_words, 6);
}

static void blake3_compress_in_place_portable(uint32_t cv[8],
                                              const uint8_t block[BLAKE3_BLOCK_LEN],
                                              uint8_t block_len, uint64_t counter,
                                              uint8_t flags) {
    uint32_t state[16];
    compress_pre(state, cv, block, block_len, counter, flags);
    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];
}

static void blake3_compress_xof_portable(const uint32_t cv[8],
                                        const uint8_t block[BLAKE3_BLOCK_LEN],
                                        uint8_t block_len, uint64_t counter,
                                        uint8_t flags, uint8_t out[64]) {
    uint32_t state[16];
    compress_pre(state, cv, block, block_len, counter, flags);

    store32(&out[ 0], state[0] ^ state[8]);
    store32(&out[ 4], state[1] ^ state[9]);
    store32(&out[ 8], state[2] ^ state[10]);
    store32(&out[12], state[3] ^ state[11]);
    store32(&out[16], state[4] ^ state[12]);
    store32(&out[20], state[5] ^ state[13]);
    store32(&out[24], state[6] ^ state[14]);
    store32(&out[28], state[7] ^ state[15]);
    store32(&out[32], state[8] ^ cv[0]);
    store32(&out[36], state[9] ^ cv[1]);
    store32(&out[40], state[10] ^ cv[2]);
    store32(&out[44], state[11] ^ cv[3]);
    store32(&out[48], state[12] ^ cv[4]);
    store32(&out[52], state[13] ^ cv[5]);
    store32(&out[56], state[14] ^ cv[6]);
    store32(&out[60], state[15] ^ cv[7]);
}

/* ---- single-compress dispatch (portable only) ---- */

static void blake3_compress_in_place(uint32_t cv[8],
                                     const uint8_t block[BLAKE3_BLOCK_LEN],
                                     uint8_t block_len, uint64_t counter,
                                     uint8_t flags) {
    blake3_compress_in_place_portable(cv, block, block_len, counter, flags);
}

static void blake3_compress_xof(const uint32_t cv[8],
                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                uint8_t block_len, uint64_t counter,
                                uint8_t flags, uint8_t out[64]) {
    blake3_compress_xof_portable(cv, block, block_len, counter, flags, out);
}

/* ---- chunk / parent / output helpers ---- */

typedef struct {
    uint32_t input_cv[8];
    uint64_t counter;
    uint8_t  block[BLAKE3_BLOCK_LEN];
    uint8_t  block_len;
    uint8_t  flags;
} blake3_output;

INLINE void chunk_state_init(blake3_chunk_state *self, const uint32_t key[8],
                             uint8_t flags) {
    memcpy(self->cv, key, BLAKE3_KEY_LEN);
    self->chunk_counter = 0;
    memset(self->buf, 0, BLAKE3_BLOCK_LEN);
    self->buf_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
}

INLINE void chunk_state_reset(blake3_chunk_state *self, const uint32_t key[8],
                              uint64_t chunk_counter) {
    memcpy(self->cv, key, BLAKE3_KEY_LEN);
    self->chunk_counter = chunk_counter;
    self->blocks_compressed = 0;
    memset(self->buf, 0, BLAKE3_BLOCK_LEN);
    self->buf_len = 0;
}

INLINE size_t chunk_state_len(const blake3_chunk_state *self) {
    return (BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed)
         + ((size_t)self->buf_len);
}

INLINE size_t chunk_state_fill_buf(blake3_chunk_state *self,
                                   const uint8_t *input, size_t input_len) {
    size_t take = BLAKE3_BLOCK_LEN - ((size_t)self->buf_len);
    if (take > input_len) take = input_len;
    memcpy(self->buf + self->buf_len, input, take);
    self->buf_len += (uint8_t)take;
    return take;
}

INLINE uint8_t chunk_state_maybe_start_flag(const blake3_chunk_state *self) {
    return (self->blocks_compressed == 0) ? CHUNK_START : 0;
}

INLINE blake3_output make_output(const uint32_t input_cv[8],
                                 const uint8_t block[BLAKE3_BLOCK_LEN],
                                 uint8_t block_len, uint64_t counter,
                                 uint8_t flags) {
    blake3_output ret;
    memcpy(ret.input_cv, input_cv, 32);
    memcpy(ret.block, block, BLAKE3_BLOCK_LEN);
    ret.block_len = block_len;
    ret.counter = counter;
    ret.flags = flags;
    return ret;
}

INLINE void output_chaining_value(const blake3_output *self, uint8_t cv[32]) {
    uint32_t cv_words[8];
    memcpy(cv_words, self->input_cv, 32);
    blake3_compress_in_place(cv_words, self->block, self->block_len,
                             self->counter, self->flags);
    store_cv_words(cv, cv_words);
}

INLINE void output_root_bytes(const blake3_output *self, uint64_t seek,
                              uint8_t *out, size_t out_len) {
    if (out_len == 0) return;
    uint64_t output_block_counter = seek / 64;
    size_t offset_within_block = (size_t)(seek % 64);
    uint8_t wide_buf[64];
    if (offset_within_block) {
        blake3_compress_xof(self->input_cv, self->block, self->block_len,
                            output_block_counter, self->flags | ROOT,
                            wide_buf);
        size_t available = 64 - offset_within_block;
        size_t bytes = out_len > available ? available : out_len;
        memcpy(out, wide_buf + offset_within_block, bytes);
        out += bytes;
        out_len -= bytes;
        output_block_counter += 1;
    }
    /* Portable: emit 64 bytes at a time using compress_xof. */
    while (out_len >= 64) {
        blake3_compress_xof(self->input_cv, self->block, self->block_len,
                            output_block_counter, self->flags | ROOT, out);
        out += 64;
        out_len -= 64;
        output_block_counter += 1;
    }
    if (out_len > 0) {
        blake3_compress_xof(self->input_cv, self->block, self->block_len,
                            output_block_counter, self->flags | ROOT, wide_buf);
        memcpy(out, wide_buf, out_len);
    }
}

INLINE void chunk_state_update(blake3_chunk_state *self, const uint8_t *input,
                               size_t input_len) {
    if (self->buf_len > 0) {
        size_t take = chunk_state_fill_buf(self, input, input_len);
        input += take;
        input_len -= take;
        if (input_len > 0) {
            blake3_compress_in_place(
                self->cv, self->buf, BLAKE3_BLOCK_LEN, self->chunk_counter,
                self->flags | chunk_state_maybe_start_flag(self));
            self->blocks_compressed += 1;
            self->buf_len = 0;
            memset(self->buf, 0, BLAKE3_BLOCK_LEN);
        }
    }
    while (input_len > BLAKE3_BLOCK_LEN) {
        blake3_compress_in_place(
            self->cv, input, BLAKE3_BLOCK_LEN, self->chunk_counter,
            self->flags | chunk_state_maybe_start_flag(self));
        self->blocks_compressed += 1;
        input += BLAKE3_BLOCK_LEN;
        input_len -= BLAKE3_BLOCK_LEN;
    }
    chunk_state_fill_buf(self, input, input_len);
}

INLINE blake3_output chunk_state_output(const blake3_chunk_state *self) {
    uint8_t block_flags =
        self->flags | chunk_state_maybe_start_flag(self) | CHUNK_END;
    return make_output(self->cv, self->buf, self->buf_len, self->chunk_counter,
                       block_flags);
}

INLINE blake3_output parent_output(const uint8_t block[BLAKE3_BLOCK_LEN],
                                   const uint32_t key[8], uint8_t flags) {
    return make_output(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
}

/* ---- public hasher ---- */

INLINE void hasher_init_base(blake3_hasher *self, const uint32_t key[8],
                             uint8_t flags) {
    memcpy(self->key, key, BLAKE3_KEY_LEN);
    chunk_state_init(&self->chunk, key, flags);
    self->cv_stack_len = 0;
}

void blake3_hasher_init(blake3_hasher *self) {
    hasher_init_base(self, IV, 0);
}

INLINE void hasher_merge_cv_stack(blake3_hasher *self, uint64_t total_chunks) {
    size_t post_merge_stack_len = (size_t)popcnt(total_chunks);
    while (self->cv_stack_len > post_merge_stack_len) {
        uint8_t *parent_node =
            &self->cv_stack[(self->cv_stack_len - 2) * BLAKE3_OUT_LEN];
        blake3_output output = parent_output(parent_node, self->key,
                                              self->chunk.flags);
        output_chaining_value(&output, parent_node);
        self->cv_stack_len -= 1;
    }
}

INLINE void hasher_push_cv(blake3_hasher *self, uint8_t new_cv[BLAKE3_OUT_LEN],
                           uint64_t chunk_counter) {
    hasher_merge_cv_stack(self, chunk_counter);
    memcpy(&self->cv_stack[self->cv_stack_len * BLAKE3_OUT_LEN], new_cv,
           BLAKE3_OUT_LEN);
    self->cv_stack_len += 1;
}

INLINE void blake3_hasher_update_base(blake3_hasher *self, const void *input,
                                      size_t input_len) {
    if (input_len == 0) return;
    const uint8_t *input_bytes = (const uint8_t *)input;

    /* If a partial chunk is buffered, fill it first. */
    if (chunk_state_len(&self->chunk) > 0) {
        size_t take = BLAKE3_CHUNK_LEN - chunk_state_len(&self->chunk);
        if (take > input_len) take = input_len;
        chunk_state_update(&self->chunk, input_bytes, take);
        input_bytes += take;
        input_len -= take;
        if (input_len > 0) {
            blake3_output output = chunk_state_output(&self->chunk);
            uint8_t chunk_cv[32];
            output_chaining_value(&output, chunk_cv);
            hasher_push_cv(self, chunk_cv, self->chunk.chunk_counter);
            chunk_state_reset(&self->chunk, self->key,
                              self->chunk.chunk_counter + 1);
        } else {
            return;
        }
    }

    /* Hash whole subtrees (power-of-two chunks) where possible. */
    while (input_len > BLAKE3_CHUNK_LEN) {
        size_t subtree_len = (size_t)round_down_to_power_of_2(input_len);
        uint64_t count_so_far = self->chunk.chunk_counter * BLAKE3_CHUNK_LEN;
        while ((((uint64_t)(subtree_len - 1)) & count_so_far) != 0) {
            subtree_len /= 2;
        }
        uint64_t subtree_chunks = subtree_len / BLAKE3_CHUNK_LEN;
        if (subtree_len <= BLAKE3_CHUNK_LEN) {
            blake3_chunk_state cs;
            chunk_state_init(&cs, self->key, self->chunk.flags);
            cs.chunk_counter = self->chunk.chunk_counter;
            chunk_state_update(&cs, input_bytes, subtree_len);
            blake3_output output = chunk_state_output(&cs);
            uint8_t cv[32];
            output_chaining_value(&output, cv);
            hasher_push_cv(self, cv, cs.chunk_counter);
        } else {
            /* Portable path: MAX_SIMD_DEGREE == 1, so we hash each chunk
             * sequentially (no SIMD parallelism available). For each pair of
             * adjacent chunks, compress_subtree_wide() would emit a parent
             * node; here we just push each chunk CV individually — the stack
             * merge logic in hasher_push_cv handles the lazy merging.
             */
            for (uint64_t c = 0; c < subtree_chunks; c++) {
                blake3_chunk_state cs;
                chunk_state_init(&cs, self->key, self->chunk.flags);
                cs.chunk_counter = self->chunk.chunk_counter + c;
                chunk_state_update(&cs, input_bytes + c * BLAKE3_CHUNK_LEN,
                                   BLAKE3_CHUNK_LEN);
                blake3_output output = chunk_state_output(&cs);
                uint8_t chunk_cv[32];
                output_chaining_value(&output, chunk_cv);
                hasher_push_cv(self, chunk_cv, cs.chunk_counter);
            }
        }
        self->chunk.chunk_counter += subtree_chunks;
        input_bytes += subtree_len;
        input_len -= subtree_len;
    }

    if (input_len > 0) {
        chunk_state_update(&self->chunk, input_bytes, input_len);
        hasher_merge_cv_stack(self, self->chunk.chunk_counter);
    }
}

void blake3_hasher_update(blake3_hasher *self, const void *input,
                          size_t input_len) {
    blake3_hasher_update_base(self, input, input_len);
}

void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out,
                            size_t out_len) {
    if (out_len == 0) return;
    /* Empty subtree stack: the current chunk (which may itself be empty) is
     * the root. Avoid the size_t underflow that would otherwise happen in
     * the else branch when both cv_stack and chunk are empty (size 0 input). */
    if (self->cv_stack_len == 0) {
        blake3_output output = chunk_state_output(&self->chunk);
        output_root_bytes(&output, 0, out, out_len);
        return;
    }
    blake3_output output;
    size_t cvs_remaining;
    if (chunk_state_len(&self->chunk) > 0) {
        cvs_remaining = self->cv_stack_len;
        output = chunk_state_output(&self->chunk);
    } else {
        cvs_remaining = self->cv_stack_len - 2;
        output = parent_output(&self->cv_stack[cvs_remaining * 32], self->key,
                               self->chunk.flags);
    }
    while (cvs_remaining > 0) {
        cvs_remaining -= 1;
        uint8_t parent_block[BLAKE3_BLOCK_LEN];
        memcpy(parent_block, &self->cv_stack[cvs_remaining * 32], 32);
        output_chaining_value(&output, &parent_block[32]);
        output = parent_output(parent_block, self->key, self->chunk.flags);
    }
    output_root_bytes(&output, 0, out, out_len);
}

void blake3_hash(const void *input, size_t input_len, uint8_t *out,
                 size_t out_len) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, input, input_len);
    blake3_hasher_finalize(&h, out, out_len);
}
