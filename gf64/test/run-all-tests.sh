#!/usr/bin/env bash
# GF64 test suite runner (issue #59 A1 verification). Builds nothing;
# expects `make -j8 all` to have been run first. Each binary's output is
# captured to /tmp/suite-<name>.log; per-binary PASS/FAIL is printed and
# SUITE_FAILED aggregates the result.
set -u
cd "$(dirname "$0")"

SUITE_FAILED=0

for t in test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita \
         test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric \
         test_gf64_mpe test_cpu_detect test_gf64_zmm_probe test_gf64_poly_mul_karatsuba \
         test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded \
         test_gf64_invmod_mod_parity test_gf64_divmod_parity test_gf64_interp_parity \
         test_gf64_additive_fft_hqc2026 test_gf64_mul_batch_shared test_gf64_region_mularr \
         test_gf64_fenger_batch test_gf64_poly_mul_internal_dispatch; do
	if [ ! -x "./$t" ]; then
		echo "MISSING: $t (build it first)"
		SUITE_FAILED=1
		continue
	fi
	if ./"$t" > "/tmp/suite-$t.log" 2>&1; then
		echo "PASS: $t"
	else
		echo "FAIL: $t (see /tmp/suite-$t.log)"
		SUITE_FAILED=1
	fi
done

echo "SUITE_FAILED=$SUITE_FAILED"
exit $SUITE_FAILED
