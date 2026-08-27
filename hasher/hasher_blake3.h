/*
 * BLAKE3 portable C implementation, ported from BLAKE3-team/BLAKE3 v1.8.7
 * (https://github.com/BLAKE3-team/BLAKE3 — Apache-2.0 / CC0).
 *
 * Stripped to the portable C core: no SSE2 / SSE4.1 / AVX2 / AVX-512 / NEON
 * dispatch. The PAR3 packet header only needs a small number of BLAKE3
 * hashes per packet (one 16-byte truncated digest over a small buffer), so
 * the portable scalar path is sufficient for correctness; throughput comes
 * from the JS-side packetizer path which already has its own BLAKE3.
 *
 * Public surface kept minimal: blake3_hasher_init / _update / _finalize.
 * We don't need keyed mode or derive-key mode for PAR3 packet checksums.
 */
#ifndef HASHER_BLAKE3_H
#define HASHER_BLAKE3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE3_OUT_LEN     32
#define BLAKE3_BLOCK_LEN   64
#define BLAKE3_CHUNK_LEN   1024
#define BLAKE3_MAX_DEPTH   54
#define BLAKE3_KEY_LEN     32

typedef struct {
    uint32_t cv[8];
    uint64_t chunk_counter;
    uint8_t  buf[BLAKE3_BLOCK_LEN];
    uint8_t  buf_len;
    uint8_t  blocks_compressed;
    uint8_t  flags;
} blake3_chunk_state;

typedef struct {
    uint32_t key[8];
    blake3_chunk_state chunk;
    uint8_t cv_stack_len;
    uint8_t cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
} blake3_hasher;

void blake3_hasher_init(blake3_hasher *self);
void blake3_hasher_update(blake3_hasher *self, const void *input, size_t input_len);
void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out, size_t out_len);

/* Convenience: one-shot BLAKE3 hash. */
void blake3_hash(const void *input, size_t input_len, uint8_t *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* HASHER_BLAKE3_H */
