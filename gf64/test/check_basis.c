#include <stdio.h>
#include <stdint.h>
typedef uint64_t gf64_t;

static const gf64_t GF64_CANTOR_BASIS[20] = {
    0x0000000000000001ULL,
    0x19c9369f278adc02ULL,
    0xa181e7d66f5ff794ULL,
    0x5db84357ce785d08ULL,
    0xb973d466f5c9d0caULL,
    0x521ac889831a075eULL,
    0x033ce8beddc8a656ULL,
    0xb5846c4e07b91010ULL,
    0x4087b8cbb37a32ecULL,
    0x00d0d3888c0ae17cULL,
    0xafd5ac70237f2222ULL,
    0xe3f5af99cc3aaaf8ULL,
    0x5a1db3b16a0b58b8ULL,
    0x09947c54fe7ee248ULL,
    0x0e8eaf0e0068f544ULL,
    0xa2a113500b4b4f5aULL,
    0xe96f9805d6ce0bb0ULL,
    0x53496f8b5c9edd4cULL,
    0xad325cb6f4ac2a9eULL,
    0x4a8dcf8bd7ede826ULL,
};

/* x^2 in GF(2^64) with x^64 + x^4 + x^3 + x + 1. */
static gf64_t gf64_sq(gf64_t a) {
    __uint128_t r = 0;
    for (int i = 0; i < 64; i++) {
        if ((a >> i) & 1ULL) r ^= (__uint128_t)1 << (2 * i);
    }
    for (int i = 127; i >= 64; i--) {
        if ((r >> i) & 1) {
            r ^= (__uint128_t)1 << i;
            r ^= (__uint128_t)1 << (i - 60);
            r ^= (__uint128_t)1 << (i - 61);
            r ^= (__uint128_t)1 << (i - 63);
            r ^= (__uint128_t)1 << (i - 64);
        }
    }
    return (gf64_t)r;
}

static gf64_t gf64_sigma(gf64_t x) { return gf64_sq(x) ^ x; }

int main(void) {
    int ok = 1;
    for (int i = 1; i < 20; i++) {
        gf64_t sig = gf64_sigma(GF64_CANTOR_BASIS[i]);
        if (sig != GF64_CANTOR_BASIS[i - 1]) {
            printf("FAIL: sigma(basis[%d]) = 0x%016llx, expected basis[%d] = 0x%016llx\n",
                   i, (unsigned long long)sig, i - 1, (unsigned long long)GF64_CANTOR_BASIS[i - 1]);
            ok = 0;
        }
    }
    if (ok) printf("All 19 basis relations hold: sigma(basis[i]) = basis[i-1] for i=1..19.\n");
    return ok ? 0 : 1;
}
