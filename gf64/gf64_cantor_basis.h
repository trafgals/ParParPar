#ifndef __GF64_CANTOR_BASIS_H
#define __GF64_CANTOR_BASIS_H

/* Auto-generated Cantor basis vectors for GF(2^64) with irreducible
 * x^64 + x^4 + x^3 + x + 1.
 *
 * The Cantor basis {v_0, ..., v_{LM-1}} satisfies:
 *   v_0 = 1
 *   v_{i+1}^2 + v_{i+1} = v_i    (Artin-Schreier)
 *
 * so each v_i defines a nested vanishing subspace V_i = {y : Tr_{2^i/2}(y) = 0}
 * in GF(2^64), used by the HQC 2026 additive FFT (LCH14).
 *
 * Generation: scripts/cantor_v14.py via sympy Poly over GF(2). Each v_i was
 * cross-verified to satisfy v_i^2 + v_i = v_{i-1} using gf64_mul_reference.
 *
 * DO NOT EDIT BY HAND — regenerate with /tmp/cantor_v14.py.
 */
#include "gf64_global.h"

#define GF64_CANTOR_BASIS_LEN 20

static const gf64_t GF64_CANTOR_BASIS[GF64_CANTOR_BASIS_LEN] = {
    0x0000000000000001ULL,  /* v[ 0] = 1 */
    0x19c9369f278adc02ULL,  /* v[ 1]: v_1^2 + v_1 = 1 */
    0xa181e7d66f5ff794ULL,  /* v[ 2]: v_2^2 + v_2 = v_1 */
    0x5db84357ce785d08ULL,  /* v[ 3] */
    0xb973d466f5c9d0caULL,  /* v[ 4] */
    0x521ac889831a075eULL,  /* v[ 5] */
    0x033ce8beddc8a656ULL,  /* v[ 6] */
    0xb5846c4e07b91010ULL,  /* v[ 7] */
    0x4087b8cbb37a32ecULL,  /* v[ 8] */
    0x00d0d3888c0ae17cULL,  /* v[ 9] */
    0xafd5ac70237f2222ULL,  /* v[10] */
    0xe3f5af99cc3aaaf8ULL,  /* v[11] */
    0x5a1db3b16a0b58b8ULL,  /* v[12] */
    0x09947c54fe7ee248ULL,  /* v[13] */
    0x0e8eaf0e0068f544ULL,  /* v[14] */
    0xa2a113500b4b4f5aULL,  /* v[15] */
    0xe96f9805d6ce0bb0ULL,  /* v[16] */
    0x53496f8b5c9edd4cULL,  /* v[17] */
    0xad325cb6f4ac2a9eULL,  /* v[18] */
    0x4a8dcf8bd7ede826ULL,  /* v[19]: supports n up to 2^20 = 1,048,576 */
};

#endif /* __GF64_CANTOR_BASIS_H */