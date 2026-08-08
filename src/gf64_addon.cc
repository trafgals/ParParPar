#include "hedley.h"

#ifdef HEDLEY_GCC_VERSION
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#include <node.h>
#include <node_buffer.h>
#include <node_version.h>
#include <v8.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#include <uv.h>
#include <js_native_api.h>
#include <node_api.h>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "gf64_global.h"
#include "par3_engine.h"
#include "platform.h"

// Forward declaration of par3_create_streaming_NAPI (defined in
// src/gf64_create_streaming.cc). Module-level export — NOT a method on
// Gf64Encoder. Registered in parpar_gf64_init_NAPI below.
extern napi_value par3_create_streaming_NAPI(napi_env env, napi_callback_info info);

using namespace v8;

class Gf64EncoderWrapper {
public:
	GF64Method method;
	static int dispatch_initialized;

	explicit Gf64EncoderWrapper(GF64Method m) : method(m) {
		if (!dispatch_initialized) {
			gf64_init_dispatch();
			dispatch_initialized = 1;
		}
	}

	inline void Mul(uint64_t* out, const uint64_t* in, size_t len, uint64_t constant) {
		gf64_region_mul(out, in, len, constant);
	}

	inline void MulArr(uint64_t* out, const uint64_t* in, const uint64_t* coeff, size_t len, size_t n_coeff) {
		gf64_region_mul_arr(out, in, coeff, len, n_coeff);
	}

	inline void CoupledMulAdd(uint64_t* out, const uint64_t* const* in, const uint64_t* coeff, size_t len, size_t G) {
		gf64_region_coupled_muladd_arr(out, (const gf64_t *HEDLEY_RESTRICT *)in, coeff, len, G);
	}

	inline void FusedOutputMulAdd(uint64_t* const* outs, const uint64_t* in, const uint64_t* const* coeff_block_starts, size_t len, size_t K) {
		gf64_region_fused_output_muladd_arr((gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)outs, (const gf64_t *HEDLEY_RESTRICT)in, (const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)coeff_block_starts, len, K);
	}

	inline void TwoDMulAdd(uint64_t* const* outs, size_t K, const uint64_t* const* in_blocks, size_t G, const uint64_t* coeff_block_2d, size_t K_stride, size_t len) {
		gf64_region_2d_muladd_arr((gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)outs, K, (const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)in_blocks, G, (const gf64_t *HEDLEY_RESTRICT)coeff_block_2d, K_stride, len);
	}
};

int Gf64EncoderWrapper::dispatch_initialized = 0;

static void Gf64EncoderWrapper_Finalize(napi_env__* env, void* data, void* hint) {
	Gf64EncoderWrapper* enc = static_cast<Gf64EncoderWrapper*>(data);
	if(enc != NULL) {
		delete enc;
	}
}

static void NAPI_CDECL Gf64EncoderWrapper_Finalize_Trampoline(napi_env__* env, void* data, void* hint) {
	Gf64EncoderWrapper_Finalize(env, data, hint);
}

static napi_value gf64_info_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 1;
	napi_value args[1];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	int method = 0;
	if(argc >= 1) {
		napi_valuetype valuetype;
		status = napi_typeof(env, args[0], &valuetype);
		if(status == napi_ok && valuetype != napi_undefined) {
			status = napi_get_value_int32(env, args[0], &method);
			if(status != napi_ok) {
				method = 0;
			}
		}
	}

	if(method == 0) {
		method = (int)gf64_detect_method();
	}

	if(method < 0 || method > 3)
		method = 3;

	napi_value ret;
	status = napi_create_object(env, &ret);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create object");
		return NULL;
	}

	const char* methodNames[] = {"AVX512", "AVX2", "SSSE3", "SCALAR"};

	napi_value method_val;
	status = napi_create_int32(env, method, &method_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create int32");
		return NULL;
	}
	status = napi_set_named_property(env, ret, "method", method_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set method property");
		return NULL;
	}

	napi_value name_val;
	status = napi_create_string_utf8(env, methodNames[method], NAPI_AUTO_LENGTH, &name_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create string");
		return NULL;
	}
	status = napi_set_named_property(env, ret, "name", name_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set name property");
		return NULL;
	}

	napi_value alignment_val;
	status = napi_create_int32(env, 64, &alignment_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create alignment int32");
		return NULL;
	}
	status = napi_set_named_property(env, ret, "alignment", alignment_val);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set alignment property");
		return NULL;
	}

	return ret;
}

static napi_value Gf64Encoder_NAPI_constructor(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 1;
	napi_value args[1];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	int method = 0;
	if(argc >= 1) {
		status = napi_get_value_int32(env, args[0], &method);
		if(status != napi_ok) {
			method = 0;
		}
	}

	if(method < 0 || method > 3) {
		method = 3;
	}

	gf64_init_dispatch();

	Gf64EncoderWrapper* enc = new Gf64EncoderWrapper((GF64Method)method);

	status = napi_wrap(env, this_arg, enc, Gf64EncoderWrapper_Finalize_Trampoline, NULL, NULL);
	if(status != napi_ok) {
		delete enc;
		napi_throw_error(env, NULL, "Failed to wrap encoder");
		return NULL;
	}

	return this_arg;
}

static napi_value Gf64Encoder_NAPI_mul(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 4;
	napi_value args[4];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 4) {
		napi_throw_type_error(env, NULL, "Requires out, in, len, and constant");
		return NULL;
	}

	Gf64EncoderWrapper* enc = NULL;
	status = napi_unwrap(env, this_arg, (void**)&enc);
	if(status != napi_ok || enc == NULL) {
		napi_throw_error(env, NULL, "Invalid encoder");
		return NULL;
	}

	uint64_t* out = NULL;
	size_t outLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&out, &outLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Output buffer required");
		return NULL;
	}
	void* aligned_out = (void*)out;
	bool needs_out_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)out & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outLen, 64)) {
			memcpy(tmp, out, outLen);
			aligned_out = tmp;
			needs_out_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	uint64_t* in = NULL;
	size_t inLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&in, &inLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Input buffer required");
		return NULL;
	}
	void* aligned_in = (void*)in;
	bool needs_in_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)in & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inLen, 64)) {
			memcpy(tmp, in, inLen);
			aligned_in = tmp;
			needs_in_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int64_t len = 0;
	status = napi_get_value_int64(env, args[2], &len);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Length must be an integer");
		return NULL;
	}

	uint64_t constant = 0;
	status = napi_get_value_int64(env, args[3], (int64_t*)&constant);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Constant must be a uint64");
		return NULL;
	}

	enc->Mul((uint64_t*)aligned_out, (uint64_t*)aligned_in, (size_t)len, constant);

	if (needs_out_temp) {
		memcpy(out, aligned_out, outLen);
		ALIGN_FREE(aligned_out);
	}
	if (needs_in_temp) {
		memcpy(in, aligned_in, inLen);
		ALIGN_FREE(aligned_in);
	}

	return NULL;
}

static napi_value Gf64Encoder_NAPI_mul_arr(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 5;
	napi_value args[5];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 5) {
		napi_throw_type_error(env, NULL, "Requires out, in, coeff, len, n_coeff");
		return NULL;
	}

	Gf64EncoderWrapper* enc = NULL;
	status = napi_unwrap(env, this_arg, (void**)&enc);
	if(status != napi_ok || enc == NULL) {
		napi_throw_error(env, NULL, "Invalid encoder");
		return NULL;
	}

	uint64_t* out = NULL;
	size_t outLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&out, &outLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Output buffer required");
		return NULL;
	}
	void* aligned_out = (void*)out;
	bool needs_out_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)out & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outLen, 64)) {
			memcpy(tmp, out, outLen);
			aligned_out = tmp;
			needs_out_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	uint64_t* in = NULL;
	size_t inLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&in, &inLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Input buffer required");
		return NULL;
	}
	void* aligned_in = (void*)in;
	bool needs_in_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)in & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inLen, 64)) {
			memcpy(tmp, in, inLen);
			aligned_in = tmp;
			needs_in_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	uint64_t* coeff = NULL;
	size_t coeffLen = 0;
	status = napi_get_buffer_info(env, args[2], (void**)&coeff, &coeffLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Coefficients buffer required");
		return NULL;
	}
	void* aligned_coeff = (void*)coeff;
	bool needs_coeff_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)coeff & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, coeffLen, 64)) {
			memcpy(tmp, coeff, coeffLen);
			aligned_coeff = tmp;
			needs_coeff_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}
	coeffLen /= sizeof(uint64_t);

	int64_t len = 0;
	status = napi_get_value_int64(env, args[3], &len);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Length must be an integer");
		return NULL;
	}

	int64_t n_coeff = 0;
	status = napi_get_value_int64(env, args[4], &n_coeff);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Num coefficients must be an integer");
		return NULL;
	}

	enc->MulArr((uint64_t*)aligned_out, (uint64_t*)aligned_in, (uint64_t*)aligned_coeff, (size_t)len, (size_t)n_coeff);

	if (needs_out_temp) {
		memcpy(out, aligned_out, outLen);
		ALIGN_FREE(aligned_out);
	}
	if (needs_in_temp) {
		memcpy(in, aligned_in, inLen);
		ALIGN_FREE(aligned_in);
	}
	if (needs_coeff_temp) {
		memcpy(coeff, aligned_coeff, coeffLen * sizeof(uint64_t));
		ALIGN_FREE(aligned_coeff);
	}

	return NULL;
}

/* Coupled-input muladd: out[w] ^= XOR_{g=0..G-1} (in[g][w] * coeff[g]).
 *
 * Arguments:
 *   args[0] (Buffer) — out destination, len * 8 bytes
 *   args[1] (Array of Buffers) — in_blocks: G input buffers, each len * 8 bytes
 *   args[2] (Buffer) — coeff: G scalars, G * 8 bytes
 *   args[3] (Number) — len: number of gf64_t elements per block
 *   args[4] (Number) — G: number of (input, coeff) pairs
 */
static napi_value Gf64Encoder_NAPI_coupled_muladd_arr(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 5;
	napi_value args[5];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 5) {
		napi_throw_type_error(env, NULL, "Requires out, in, coeff, len, G");
		return NULL;
	}

	Gf64EncoderWrapper* enc = NULL;
	status = napi_unwrap(env, this_arg, (void**)&enc);
	if(status != napi_ok || enc == NULL) {
		napi_throw_error(env, NULL, "Invalid encoder");
		return NULL;
	}

	uint64_t* out = NULL;
	size_t outLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&out, &outLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Output buffer required");
		return NULL;
	}
	void* aligned_out = (void*)out;
	bool needs_out_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)out & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outLen, 64)) {
			memcpy(tmp, out, outLen);
			aligned_out = tmp;
			needs_out_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int64_t G_signed = 0;
	status = napi_get_value_int64(env, args[4], &G_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "G must be an integer");
		return NULL;
	}
	if(G_signed < 0 || G_signed > 65536) {
		napi_throw_range_error(env, NULL, "G out of range [0, 65536]");
		return NULL;
	}
	size_t G = (size_t)G_signed;

	int64_t len = 0;
	status = napi_get_value_int64(env, args[3], &len);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Length must be an integer");
		return NULL;
	}
	if(len < 0) {
		napi_throw_range_error(env, NULL, "Length must be non-negative");
		return NULL;
	}

	uint64_t* coeff = NULL;
	size_t coeffLen = 0;
	status = napi_get_buffer_info(env, args[2], (void**)&coeff, &coeffLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Coefficients buffer required");
		return NULL;
	}
	void* aligned_coeff = (void*)coeff;
	bool needs_coeff_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)coeff & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, coeffLen, 64)) {
			memcpy(tmp, coeff, coeffLen);
			aligned_coeff = tmp;
			needs_coeff_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}
	if(coeffLen < G * sizeof(uint64_t)) {
		napi_throw_range_error(env, NULL, "Coefficients buffer too small for G");
		return NULL;
	}

	bool is_array = false;
	status = napi_is_array(env, args[1], &is_array);
	if(status != napi_ok || !is_array) {
		napi_throw_type_error(env, NULL, "Inputs must be an array of buffers");
		return NULL;
	}

	uint32_t in_len_u32 = 0;
	status = napi_get_array_length(env, args[1], &in_len_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get input array length");
		return NULL;
	}
	if((size_t)in_len_u32 != G) {
		napi_throw_range_error(env, NULL, "Input array length must equal G");
		return NULL;
	}

	/* Collect per-block pointers. For large G (> 1024) we fall back to a
	 * heap-allocated pointer array so we don't blow the C stack. */
	const uint64_t* in_blocks_stack[1024];
	const uint64_t** in_blocks;
	bool heap_allocated = false;
	if(G <= 1024) {
		in_blocks = in_blocks_stack;
	} else {
		in_blocks = (const uint64_t**)malloc(G * sizeof(const uint64_t*));
		if(in_blocks == NULL) {
			napi_throw_error(env, NULL, "Out of memory allocating in_blocks");
			return NULL;
		}
		heap_allocated = true;
	}

	for(uint32_t g = 0; g < in_len_u32; g++) {
		napi_value elem;
		status = napi_get_element(env, args[1], g, &elem);
		if(status != napi_ok) {
			if(heap_allocated) free(in_blocks);
			napi_throw_error(env, NULL, "Failed to read input array element");
			return NULL;
		}
		uint64_t* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, (void**)&p, &pl);
		if(status != napi_ok) {
			if(heap_allocated) free(in_blocks);
			napi_throw_type_error(env, NULL, "Input array element must be a Buffer");
			return NULL;
		}
		in_blocks[g] = p;
	}

	enc->CoupledMulAdd((uint64_t*)aligned_out, in_blocks, (uint64_t*)aligned_coeff, (size_t)len, G);

	if(heap_allocated) free(in_blocks);

	if (needs_out_temp) {
		memcpy(out, aligned_out, outLen);
		ALIGN_FREE(aligned_out);
	}
	if (needs_coeff_temp) {
		memcpy(coeff, aligned_coeff, coeffLen);
		ALIGN_FREE(aligned_coeff);
	}

	return NULL;
}

/* Fused-output muladd: for each k in [0..K): outs[k][w] ^= in[w] * coeff_block_starts[k]
 * for all w in [0..len). All K outputs are updated from a single input block.
 *
 * Arguments:
 *   args[0] (Array of Buffers) — outs: K destination buffers, each len * 8 bytes
 *   args[1] (Buffer) — in: single input block, len * 8 bytes
 *   args[2] (Array of Buffers) — coeff_block_starts: K scalar buffers, each 8 bytes
 *   args[3] (Number) — len: number of gf64_t elements per output block
 *   args[4] (Number) — K: number of (output, coeff) pairs
 *
 * Coefficient layout choice: an Array<K> of single-scalar Buffers (mirrors
 * the coupled_muladd_arr in-blocks pattern). This keeps the JS-side surface
 * symmetric with coupled_muladd_arr — both new bindings use Array<K> of
 * per-element Buffers for their pointer-array marshaling. A flat Buffer of
 * K scalars would also satisfy the in-pointer-array dispatch, but would
 * diverge from the established per-element-pointer pattern.
 */
static napi_value Gf64Encoder_NAPI_fused_output_muladd_arr(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 5;
	napi_value args[5];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 5) {
		napi_throw_type_error(env, NULL, "Requires outs, in, coeff, len, K");
		return NULL;
	}

	Gf64EncoderWrapper* enc = NULL;
	status = napi_unwrap(env, this_arg, (void**)&enc);
	if(status != napi_ok || enc == NULL) {
		napi_throw_error(env, NULL, "Invalid encoder");
		return NULL;
	}

	int64_t K_signed = 0;
	status = napi_get_value_int64(env, args[4], &K_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "K must be an integer");
		return NULL;
	}
	if(K_signed < 0 || K_signed > 65536) {
		napi_throw_range_error(env, NULL, "K out of range [0, 65536]");
		return NULL;
	}
	size_t K = (size_t)K_signed;

	int64_t len = 0;
	status = napi_get_value_int64(env, args[3], &len);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Length must be an integer");
		return NULL;
	}
	if(len < 0) {
		napi_throw_range_error(env, NULL, "Length must be non-negative");
		return NULL;
	}

	const uint64_t* in = NULL;
	size_t inLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&in, &inLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Input buffer required");
		return NULL;
	}
	void* aligned_in = (void*)in;
	bool needs_in_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)in & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inLen, 64)) {
			memcpy(tmp, in, inLen);
			aligned_in = tmp;
			needs_in_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}
	if(inLen < (size_t)len * sizeof(uint64_t)) {
		napi_throw_range_error(env, NULL, "Input buffer too small for len");
		return NULL;
	}

	bool outs_is_array = false;
	status = napi_is_array(env, args[0], &outs_is_array);
	if(status != napi_ok || !outs_is_array) {
		napi_throw_type_error(env, NULL, "Outputs must be an array of buffers");
		return NULL;
	}

	uint32_t outs_len_u32 = 0;
	status = napi_get_array_length(env, args[0], &outs_len_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get outputs array length");
		return NULL;
	}
	if((size_t)outs_len_u32 != K) {
		napi_throw_range_error(env, NULL, "Outputs array length must equal K");
		return NULL;
	}

	bool coeff_is_array = false;
	status = napi_is_array(env, args[2], &coeff_is_array);
	if(status != napi_ok || !coeff_is_array) {
		napi_throw_type_error(env, NULL, "Coefficients must be an array of buffers");
		return NULL;
	}

	uint32_t coeff_len_u32 = 0;
	status = napi_get_array_length(env, args[2], &coeff_len_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get coefficients array length");
		return NULL;
	}
	if((size_t)coeff_len_u32 != K) {
		napi_throw_range_error(env, NULL, "Coefficients array length must equal K");
		return NULL;
	}

	/* Collect per-block pointers. Stack-allocated for K<=1024; heap fallback
	 * for larger K to avoid blowing the C stack. Two parallel arrays share
	 * the same lifetime, so we use a single allocation size. */
	uint64_t* outs_stack[1024];
	const uint64_t* coeff_stack[1024];
	uint64_t** outs_ptrs;
	const uint64_t** coeff_ptrs;
	bool heap_allocated = false;
	if(K <= 1024) {
		outs_ptrs = outs_stack;
		coeff_ptrs = coeff_stack;
	} else {
		outs_ptrs = (uint64_t**)malloc(K * sizeof(uint64_t*));
		coeff_ptrs = (const uint64_t**)malloc(K * sizeof(const uint64_t*));
		if(outs_ptrs == NULL || coeff_ptrs == NULL) {
			if(outs_ptrs != NULL) free(outs_ptrs);
			if(coeff_ptrs != NULL) free(coeff_ptrs);
			napi_throw_error(env, NULL, "Out of memory allocating pointer arrays");
			return NULL;
		}
		heap_allocated = true;
	}

	for(uint32_t k = 0; k < outs_len_u32; k++) {
		napi_value elem;
		status = napi_get_element(env, args[0], k, &elem);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_error(env, NULL, "Failed to read outputs array element");
			return NULL;
		}
		uint64_t* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, (void**)&p, &pl);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_type_error(env, NULL, "Outputs array element must be a Buffer");
			return NULL;
		}
		if(pl < (size_t)len * sizeof(uint64_t)) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_range_error(env, NULL, "Outputs array element buffer too small for len");
			return NULL;
		}
		outs_ptrs[k] = p;
	}

	for(uint32_t k = 0; k < coeff_len_u32; k++) {
		napi_value elem;
		status = napi_get_element(env, args[2], k, &elem);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_error(env, NULL, "Failed to read coefficients array element");
			return NULL;
		}
		const uint64_t* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, (void**)&p, &pl);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_type_error(env, NULL, "Coefficients array element must be a Buffer");
			return NULL;
		}
		if(pl < sizeof(uint64_t)) {
			if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }
			napi_throw_range_error(env, NULL, "Coefficients array element buffer too small for one scalar");
			return NULL;
		}
		coeff_ptrs[k] = p;
	}

	enc->FusedOutputMulAdd(outs_ptrs, (const uint64_t*)aligned_in, coeff_ptrs, (size_t)len, K);

	if(heap_allocated) { free(outs_ptrs); free(coeff_ptrs); }

	if (needs_in_temp) {
		memcpy((void*)in, aligned_in, inLen);
		ALIGN_FREE(aligned_in);
	}

	return NULL;
}

/* 2D-blocked muladd: for each (k, g) in [0..K) x [0..G):
 *   outs[k][w] ^= in_blocks[g][w] * coeff_block_2d[k * K_stride + g]
 * for all w in [0..len). This is the most memory-efficient of the three
 * new muladd variants (K outputs x G inputs per call) — it reduces both
 * input load count AND output store count compared to a coupled-only or
 * fused-only path.
 *
 * Arguments:
 *   args[0] (Array of Buffers) — outs: K destination buffers, each len * 8 bytes
 *   args[1] (Number) — K: number of output blocks in the tile
 *   args[2] (Array of Buffers) — in_blocks: G input buffers, each len * 8 bytes
 *   args[3] (Number) — G: number of input blocks in the tile
 *   args[4] (Buffer) — coeff_block_2d: 2D coefficient matrix, K * K_stride * 8 bytes
 *                      (laid out row-major: coeff[k * K_stride + g] for k in [0..K), g in [0..G)).
 *                      K_stride may exceed G (padding to SIMD boundary) — caller is responsible
 *                      for zero-padding the unused tail columns of each row.
 *   args[5] (Number) — K_stride: row stride for the 2D coefficient matrix; must be >= G
 *   args[6] (Number) — len: number of gf64_t elements per block
 */
static napi_value Gf64Encoder_NAPI_two_d_muladd_arr(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 7;
	napi_value args[7];
	napi_value this_arg;

	status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL, "Requires outs, K, in_blocks, G, coeff_block_2d, K_stride, len");
		return NULL;
	}

	Gf64EncoderWrapper* enc = NULL;
	status = napi_unwrap(env, this_arg, (void**)&enc);
	if(status != napi_ok || enc == NULL) {
		napi_throw_error(env, NULL, "Invalid encoder");
		return NULL;
	}

	int64_t len_signed = 0;
	status = napi_get_value_int64(env, args[6], &len_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "len must be an integer");
		return NULL;
	}
	if(len_signed < 0) {
		napi_throw_range_error(env, NULL, "len must be non-negative");
		return NULL;
	}
	size_t len = (size_t)len_signed;

	int64_t K_signed = 0;
	status = napi_get_value_int64(env, args[1], &K_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "K must be an integer");
		return NULL;
	}
	if(K_signed < 0 || K_signed > 65536) {
		napi_throw_range_error(env, NULL, "K out of range [0, 65536]");
		return NULL;
	}
	size_t K = (size_t)K_signed;

	int64_t G_signed = 0;
	status = napi_get_value_int64(env, args[3], &G_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "G must be an integer");
		return NULL;
	}
	if(G_signed < 0 || G_signed > 65536) {
		napi_throw_range_error(env, NULL, "G out of range [0, 65536]");
		return NULL;
	}
	size_t G = (size_t)G_signed;

	int64_t K_stride_signed = 0;
	status = napi_get_value_int64(env, args[5], &K_stride_signed);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "K_stride must be an integer");
		return NULL;
	}
	if(K_stride_signed < (int64_t)G) {
		napi_throw_range_error(env, NULL, "K_stride must be >= G");
		return NULL;
	}
	if(K_stride_signed > 65536) {
		napi_throw_range_error(env, NULL, "K_stride out of range [0, 65536]");
		return NULL;
	}
	size_t K_stride = (size_t)K_stride_signed;

	const uint64_t* coeff_block_2d = NULL;
	size_t coeffLen = 0;
	status = napi_get_buffer_info(env, args[4], (void**)&coeff_block_2d, &coeffLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "coeff_block_2d buffer required");
		return NULL;
	}
	void* aligned_coeff_block_2d = (void*)coeff_block_2d;
	bool needs_coeff_block_2d_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)coeff_block_2d & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, coeffLen, 64)) {
			memcpy(tmp, coeff_block_2d, coeffLen);
			aligned_coeff_block_2d = tmp;
			needs_coeff_block_2d_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}
	if(coeffLen < K * K_stride * sizeof(uint64_t)) {
		napi_throw_range_error(env, NULL, "coeff_block_2d buffer too small for K * K_stride");
		return NULL;
	}

	bool outs_is_array = false;
	status = napi_is_array(env, args[0], &outs_is_array);
	if(status != napi_ok || !outs_is_array) {
		napi_throw_type_error(env, NULL, "outs must be an array of buffers");
		return NULL;
	}

	uint32_t outs_len_u32 = 0;
	status = napi_get_array_length(env, args[0], &outs_len_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get outs array length");
		return NULL;
	}
	if((size_t)outs_len_u32 != K) {
		napi_throw_range_error(env, NULL, "outs array length must equal K");
		return NULL;
	}

	bool in_is_array = false;
	status = napi_is_array(env, args[2], &in_is_array);
	if(status != napi_ok || !in_is_array) {
		napi_throw_type_error(env, NULL, "in_blocks must be an array of buffers");
		return NULL;
	}

	uint32_t in_len_u32 = 0;
	status = napi_get_array_length(env, args[2], &in_len_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get in_blocks array length");
		return NULL;
	}
	if((size_t)in_len_u32 != G) {
		napi_throw_range_error(env, NULL, "in_blocks array length must equal G");
		return NULL;
	}

	/* Collect per-block pointers. Stack-allocated for K+G<=1024; heap fallback
	 * for larger combined size to avoid blowing the C stack. Two parallel arrays
	 * share the same lifetime, so we use a single allocation. */
	uint64_t* outs_stack[1024];
	const uint64_t* in_stack[1024];
	uint64_t** outs_ptrs;
	const uint64_t** in_ptrs;
	bool heap_allocated = false;
	if(K <= 1024 && G <= 1024) {
		outs_ptrs = outs_stack;
		in_ptrs = in_stack;
	} else {
		outs_ptrs = (uint64_t**)malloc(K * sizeof(uint64_t*));
		in_ptrs = (const uint64_t**)malloc(G * sizeof(const uint64_t*));
		if(outs_ptrs == NULL || in_ptrs == NULL) {
			if(outs_ptrs != NULL) free(outs_ptrs);
			if(in_ptrs != NULL) free(in_ptrs);
			napi_throw_error(env, NULL, "Out of memory allocating pointer arrays");
			return NULL;
		}
		heap_allocated = true;
	}

	for(uint32_t k = 0; k < outs_len_u32; k++) {
		napi_value elem;
		status = napi_get_element(env, args[0], k, &elem);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_error(env, NULL, "Failed to read outs array element");
			return NULL;
		}
		uint64_t* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, (void**)&p, &pl);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_type_error(env, NULL, "outs array element must be a Buffer");
			return NULL;
		}
		if(pl < (size_t)len * sizeof(uint64_t)) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_range_error(env, NULL, "outs array element buffer too small for len");
			return NULL;
		}
		outs_ptrs[k] = p;
	}

	for(uint32_t g = 0; g < in_len_u32; g++) {
		napi_value elem;
		status = napi_get_element(env, args[2], g, &elem);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_error(env, NULL, "Failed to read in_blocks array element");
			return NULL;
		}
		const uint64_t* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, (void**)&p, &pl);
		if(status != napi_ok) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_type_error(env, NULL, "in_blocks array element must be a Buffer");
			return NULL;
		}
		if(pl < (size_t)len * sizeof(uint64_t)) {
			if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }
			napi_throw_range_error(env, NULL, "in_blocks array element buffer too small for len");
			return NULL;
		}
		in_ptrs[g] = p;
	}

	enc->TwoDMulAdd(outs_ptrs, K, in_ptrs, G, (uint64_t*)aligned_coeff_block_2d, K_stride, len);

	if(heap_allocated) { free(outs_ptrs); free(in_ptrs); }

	if (needs_coeff_block_2d_temp) {
		memcpy((void*)coeff_block_2d, aligned_coeff_block_2d, coeffLen);
		ALIGN_FREE(aligned_coeff_block_2d);
	}

	return NULL;
}

// Factory function for par3gen.js compatibility
static napi_value Gf64Encoder_NAPI_create(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 2;
	napi_value args[2];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	int method = 0;
	if(argc >= 1) {
		status = napi_get_value_int32(env, args[0], &method);
		if(status != napi_ok) {
			method = 0;
		}
	}

	int numThreads = 0;
	if(argc >= 2) {
		status = napi_get_value_int32(env, args[1], &numThreads);
		if(status != napi_ok) {
			numThreads = 0;
		}
	}

	gf64_init_dispatch();

	Gf64EncoderWrapper* enc = new Gf64EncoderWrapper((GF64Method)method);
	(void)numThreads; // Currently unused but part of API

	napi_value result;
	// Don't set a finalizer here - Gf64Encoder_NAPI_destroy is solely responsible for cleanup
	// to avoid double-free when destroy is explicitly called
	status = napi_create_external(env, enc, NULL, NULL, &result);
	if(status != napi_ok) {
		delete enc;
		napi_throw_error(env, NULL, "Failed to create external");
		return NULL;
	}

	return result;
}

static napi_value Gf64Encoder_NAPI_destroy(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 1;
	napi_value args[1];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	void* data = NULL;
	status = napi_get_value_external(env, args[0], &data);
	if(status != napi_ok || data == NULL) {
		napi_throw_error(env, NULL, "Invalid external value");
		return NULL;
	}

	// Just delete directly - Finalize would do the same
	delete (Gf64EncoderWrapper*)data;

	return NULL;
}

extern "C" int gf64_solve(gf64_t* A, gf64_t* b, gf64_t* x, size_t n);

static napi_value gf64_solve_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 3;
	napi_value args[3];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 3) {
		napi_throw_type_error(env, NULL, "Requires A, b, and n");
		return NULL;
	}

	gf64_t* A = NULL;
	size_t ALen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&A, &ALen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Matrix A buffer required");
		return NULL;
	}
	void* aligned_A = (void*)A;
	bool needs_A_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)A & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, ALen, 64)) {
			memcpy(tmp, A, ALen);
			aligned_A = tmp;
			needs_A_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* b = NULL;
	size_t bLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&b, &bLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "Vector b buffer required");
		return NULL;
	}
	void* aligned_b = (void*)b;
	bool needs_b_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)b & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, bLen, 64)) {
			memcpy(tmp, b, bLen);
			aligned_b = tmp;
			needs_b_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t n = 0;
	status = napi_get_value_int32(env, args[2], &n);
	if(status != napi_ok || n <= 0) {
		napi_throw_type_error(env, NULL, "Dimension n must be positive integer");
		return NULL;
	}

	size_t nSize = (size_t)n;
	if(ALen < nSize * nSize * sizeof(gf64_t)) {
		napi_throw_error(env, NULL, "Matrix A buffer too small");
		return NULL;
	}
	if(bLen < nSize * sizeof(gf64_t)) {
		napi_throw_error(env, NULL, "Vector b buffer too small");
		return NULL;
	}

	gf64_t* x = (gf64_t*)malloc(nSize * sizeof(gf64_t));
	if(!x) {
		napi_throw_error(env, NULL, "Failed to allocate solution vector");
		return NULL;
	}

	gf64_init_dispatch();

	int result = gf64_solve((gf64_t*)aligned_A, (gf64_t*)aligned_b, x, nSize);

	napi_value result_val;
	if(result == 0) {
		status = napi_create_buffer_copy(env, nSize * sizeof(gf64_t), x, NULL, &result_val);
	} else {
		status = napi_get_undefined(env, &result_val);
	}

	free(x);

	if (needs_A_temp) {
		memcpy(A, aligned_A, nSize * nSize * sizeof(gf64_t));
		ALIGN_FREE(aligned_A);
	}
	if (needs_b_temp) {
		memcpy(b, aligned_b, nSize * sizeof(gf64_t));
		ALIGN_FREE(aligned_b);
	}

	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create result");
		return NULL;
	}

	return result_val;
}

// Helper: extract uint64 from napi_value (accepts Number or BigInt).
// Extract uint64 from napi_value (Number or BigInt).
// Tries BigInt first (via napi_get_value_bigint_uint64), then falls back to Number.
// Returns napi_ok on success, napi_generic_failure if neither type.
static napi_status get_uint64_from_value(napi_env env, napi_value val, uint64_t* result) {
	napi_status status;
	napi_valuetype valuetype;

	status = napi_typeof(env, val, &valuetype);
	if(status != napi_ok) return status;

	if(valuetype == napi_bigint) {
		bool lossless = false;
		status = napi_get_value_bigint_uint64(env, val, result, &lossless);
		if(status == napi_ok) return status;
		// Fall through to Number attempt (rare — only if BigInt conversion fails)
	}

	int64_t tmp;
	status = napi_get_value_int64(env, val, &tmp);
	if(status == napi_ok) {
		*result = (uint64_t)tmp;
		return napi_ok;
	}

	return napi_generic_failure;
}

static napi_value ComputeRecovery_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 8;
	napi_value args[8];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL, "Requires inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	// Extract buffer args
	gf64_t* inputs = NULL;
	size_t inputsLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&inputs, &inputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "inputs must be a Buffer");
		return NULL;
	}
	void* aligned_inputs = (void*)inputs;
	bool needs_inputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)inputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inputsLen, 64)) {
			memcpy(tmp, inputs, inputsLen);
			aligned_inputs = tmp;
			needs_inputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&outputs, &outputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "outputs must be a Buffer");
		return NULL;
	}
	void* aligned_outputs = (void*)outputs;
	bool needs_outputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outputsLen, 64)) {
			memcpy(tmp, outputs, outputsLen);
			aligned_outputs = tmp;
			needs_outputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	// Extract integer args
	int32_t numInputs = 0;
	status = napi_get_value_int32(env, args[2], &numInputs);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numInputs must be an integer");
		return NULL;
	}

	int32_t numRecovery = 0;
	status = napi_get_value_int32(env, args[3], &numRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numRecovery must be an integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	// Parse firstInput (Number or BigInt)
	uint64_t firstInput = 0;
	status = get_uint64_from_value(env, args[5], &firstInput);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt");
		return NULL;
	}

	// Parse firstRecovery (Number or BigInt)
	uint64_t firstRecovery = 0;
	status = get_uint64_from_value(env, args[6], &firstRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt");
		return NULL;
	}

	// Parse numThreads (optional, defaults to 0 = auto)
	int32_t numThreads = 0;
	if(argc >= 8) {
		status = napi_get_value_int32(env, args[7], &numThreads);
		if(status != napi_ok) {
			napi_throw_type_error(env, NULL, "numThreads must be an integer");
			return NULL;
		}
	}

	// Validation
	if(numInputs <= 0) {
		napi_throw_range_error(env, NULL, "numInputs must be positive");
		return NULL;
	}

	if(numRecovery <= 0) {
		napi_throw_range_error(env, NULL, "numRecovery must be positive");
		return NULL;
	}

	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	if(inputsLen < (size_t)(numInputs * blockSize)) {
		napi_throw_range_error(env, NULL, "inputs buffer too small for numInputs * blockSize");
		return NULL;
	}

	if(outputsLen < (size_t)(numRecovery * blockSize)) {
		napi_throw_range_error(env, NULL, "outputs buffer too small for numRecovery * blockSize");
		return NULL;
	}

	// Convert blockSize (bytes) to blockSize64 (64-bit words)
	size_t blockSize64 = (size_t)(blockSize / 8);

	// Call the engine
	gf64_init_dispatch();
	GF64Controller::ComputeRecoveryBlocks(
		(gf64_t*)aligned_inputs, (size_t)numInputs,
		(gf64_t*)aligned_outputs, (size_t)numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		(int)numThreads
	);

	if (needs_inputs_temp) {
		memcpy(inputs, aligned_inputs, (size_t)numInputs * blockSize);
		ALIGN_FREE(aligned_inputs);
	}
	if (needs_outputs_temp) {
		memcpy(outputs, aligned_outputs, (size_t)numRecovery * blockSize);
		ALIGN_FREE(aligned_outputs);
	}

	return NULL;
}

// compute_recovery_full NAPI binding — single-call version of compute_recovery.
// Args: inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery, numThreads
// Same args as ComputeRecovery_NAPI. The "full" variant routes through the
// single-call engine entry GF64Controller::ComputeRecoveryBlocksFull, which
// builds the full Cauchy matrix once via the LRU cache and runs the L3-aware
// WorkerThread over the full recovery range in one shot. This is the path
// that lib/par3gen.js's create-path uses when PAR3_BATCH_SIZE is unset; T8's
// lib/par3gen.js refactor is what eliminates the JS-side per-batch call
// pattern, not this NAPI binding.
static napi_value ComputeRecoveryFull_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 8;
	napi_value args[8];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL, "Requires inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	gf64_t* inputs = NULL;
	size_t inputsLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&inputs, &inputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "inputs must be a Buffer");
		return NULL;
	}
	void* aligned_inputs = (void*)inputs;
	bool needs_inputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)inputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inputsLen, 64)) {
			memcpy(tmp, inputs, inputsLen);
			aligned_inputs = tmp;
			needs_inputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&outputs, &outputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "outputs must be a Buffer");
		return NULL;
	}
	void* aligned_outputs = (void*)outputs;
	bool needs_outputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outputsLen, 64)) {
			memcpy(tmp, outputs, outputsLen);
			aligned_outputs = tmp;
			needs_outputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t numInputs = 0;
	status = napi_get_value_int32(env, args[2], &numInputs);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numInputs must be an integer");
		return NULL;
	}

	int32_t numRecovery = 0;
	status = napi_get_value_int32(env, args[3], &numRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numRecovery must be an integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	uint64_t firstInput = 0;
	status = get_uint64_from_value(env, args[5], &firstInput);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt");
		return NULL;
	}

	uint64_t firstRecovery = 0;
	status = get_uint64_from_value(env, args[6], &firstRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt");
		return NULL;
	}

	int32_t numThreads = 0;
	if(argc >= 8) {
		status = napi_get_value_int32(env, args[7], &numThreads);
		if(status != napi_ok) {
			napi_throw_type_error(env, NULL, "numThreads must be an integer");
			return NULL;
		}
	}

	if(numInputs <= 0) {
		napi_throw_range_error(env, NULL, "numInputs must be positive");
		return NULL;
	}

	if(numRecovery <= 0) {
		napi_throw_range_error(env, NULL, "numRecovery must be positive");
		return NULL;
	}

	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	if(inputsLen < (size_t)(numInputs * blockSize)) {
		napi_throw_range_error(env, NULL, "inputs buffer too small for numInputs * blockSize");
		return NULL;
	}

	if(outputsLen < (size_t)(numRecovery * blockSize)) {
		napi_throw_range_error(env, NULL, "outputs buffer too small for numRecovery * blockSize");
		return NULL;
	}

	size_t blockSize64 = (size_t)(blockSize / 8);

	gf64_init_dispatch();
	long long t0 = 0, t1 = 0;
	if (getenv("PAR3_PROFILE")) {
#ifndef _WIN32
		struct timeval tv0;
		gettimeofday(&tv0, NULL);
		t0 = (long long)tv0.tv_sec * 1000000LL + (long long)tv0.tv_usec;
#endif
	}

GF64Controller::ComputeRecoveryBlocksFull(
		(gf64_t*)aligned_inputs, (size_t)numInputs,
		(gf64_t*)aligned_outputs, (size_t)numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		(int)numThreads
	);

	// v2: per-stage kernel timing. PAR3_PROFILE must be set in the parent
	// process env (the C side reads it via getenv since the C env is
	// shared). Output goes to stderr so it survives stdout buffering in
	// pipe-attached child processes.
	if (getenv("PAR3_PROFILE")) {
#ifndef _WIN32
		struct timeval tv2;
		gettimeofday(&tv2, NULL);
		t1 = (long long)tv2.tv_sec * 1000000LL + (long long)tv2.tv_usec;
		fprintf(stderr, "[PAR3_PROFILE-KERNEL] numInputs=%zu numRecovery=%zu blockSize64=%zu single-call took=%lldus\n",
			(size_t)numInputs, (size_t)numRecovery, (size_t)blockSize64, t1 - t0);
#else
		fprintf(stderr, "[PAR3_PROFILE-KERNEL] numInputs=%zu numRecovery=%zu blockSize64=%zu single-call\n",
			(size_t)numInputs, (size_t)numRecovery, (size_t)blockSize64);
#endif
	}

	if (needs_inputs_temp) {
		memcpy(inputs, aligned_inputs, (size_t)numInputs * blockSize);
		ALIGN_FREE(aligned_inputs);
	}
	if (needs_outputs_temp) {
		memcpy(outputs, aligned_outputs, (size_t)numRecovery * blockSize);
		ALIGN_FREE(aligned_outputs);
	}

	return NULL;
}

// compute_recovery_barycentric NAPI binding — barycentric-form single-call
// variant of compute_recovery_full. Same args as ComputeRecoveryFull_NAPI,
// but routes through GF64Controller::ComputeRecoveryBlocksBarycentric, which
// uses the barycentric Lagrange interpolation form (subproduct tree + point
// evaluation) instead of building the dense Cauchy matrix. This is the path
// T9 (par3_engine_barycentric.cc) opens up; lib/par3gen.js exposes this as
// the recovery-side entry that avoids the O(N²) matrix-build cost.
static napi_value ComputeRecoveryBarycentric_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 8;
	napi_value args[8];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL, "Requires inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	gf64_t* inputs = NULL;
	size_t inputsLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&inputs, &inputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "inputs must be a Buffer");
		return NULL;
	}
	void* aligned_inputs = (void*)inputs;
	bool needs_inputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)inputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inputsLen, 64)) {
			memcpy(tmp, inputs, inputsLen);
			aligned_inputs = tmp;
			needs_inputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&outputs, &outputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "outputs must be a Buffer");
		return NULL;
	}
	void* aligned_outputs = (void*)outputs;
	bool needs_outputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outputsLen, 64)) {
			memcpy(tmp, outputs, outputsLen);
			aligned_outputs = tmp;
			needs_outputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t numInputs = 0;
	status = napi_get_value_int32(env, args[2], &numInputs);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numInputs must be an integer");
		return NULL;
	}

	int32_t numRecovery = 0;
	status = napi_get_value_int32(env, args[3], &numRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numRecovery must be an integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	uint64_t firstInput = 0;
	status = get_uint64_from_value(env, args[5], &firstInput);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt");
		return NULL;
	}

	uint64_t firstRecovery = 0;
	status = get_uint64_from_value(env, args[6], &firstRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt");
		return NULL;
	}

	int32_t numThreads = 0;
	if(argc >= 8) {
		status = napi_get_value_int32(env, args[7], &numThreads);
		if(status != napi_ok) {
			napi_throw_type_error(env, NULL, "numThreads must be an integer");
			return NULL;
		}
	}

	if(numInputs <= 0) {
		napi_throw_range_error(env, NULL, "numInputs must be positive");
		return NULL;
	}

	if(numRecovery <= 0) {
		napi_throw_range_error(env, NULL, "numRecovery must be positive");
		return NULL;
	}

	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	if(inputsLen < (size_t)(numInputs * blockSize)) {
		napi_throw_range_error(env, NULL, "inputs buffer too small for numInputs * blockSize");
		return NULL;
	}

	if(outputsLen < (size_t)(numRecovery * blockSize)) {
		napi_throw_range_error(env, NULL, "outputs buffer too small for numRecovery * blockSize");
		return NULL;
	}

	size_t blockSize64 = (size_t)(blockSize / 8);

	gf64_init_dispatch();

	GF64Controller::ComputeRecoveryBlocksBarycentric(
		(const gf64_t*)aligned_inputs, (size_t)numInputs,
		(gf64_t*)aligned_outputs, (size_t)numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		(int)numThreads
	);

	if (needs_inputs_temp) {
		memcpy(inputs, aligned_inputs, (size_t)numInputs * blockSize);
		ALIGN_FREE(aligned_inputs);
	}
	if (needs_outputs_temp) {
		memcpy(outputs, aligned_outputs, (size_t)numRecovery * blockSize);
		ALIGN_FREE(aligned_outputs);
	}

	return NULL;
}

// compute_recovery_fenger NAPI binding — Fenger-Toeplitz (issue #28) single-
// call variant of compute_recovery_full. Same args as ComputeRecoveryFull_NAPI,
// but routes through GF64Controller::ComputeRecoveryBlocksFenger, which uses
// the Bostan-Schost top-down Fenger pipeline from gf64/gf64_fenger.c.
//
// Constraints (forwarded from the engine entry):
//   - numInputs and numRecovery must each be 0, 1, or a power of 2 (the
//     subproduct-tree builder in gf64_subproduct.c requires it).
//   - The engine entry falls back to the legacy 2D-muladd path when the
//     constraint is not met, so the NAPI binding is safe to call from any
//     JS-side caller — but for the canonical PAR3 workload (powers of 2
//     slice counts) it routes through the Fenger pipeline.
static napi_value ComputeRecoveryFenger_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 8;
	napi_value args[8];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL, "Requires inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	gf64_t* inputs = NULL;
	size_t inputsLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&inputs, &inputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "inputs must be a Buffer");
		return NULL;
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&outputs, &outputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "outputs must be a Buffer");
		return NULL;
	}

	int64_t numInputs = 0, numRecovery = 0;
	status = napi_get_value_int64(env, args[2], &numInputs);
	if(status != napi_ok) { napi_throw_type_error(env, NULL, "numInputs must be an integer"); return NULL; }
	status = napi_get_value_int64(env, args[3], &numRecovery);
	if(status != napi_ok) { napi_throw_type_error(env, NULL, "numRecovery must be an integer"); return NULL; }
	if(numInputs < 0 || numRecovery < 0) {
		napi_throw_range_error(env, NULL, "numInputs and numRecovery must be non-negative");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	uint64_t firstInput = 0, firstRecovery = 0;
	status = get_uint64_from_value(env, args[5], &firstInput);
	if(status != napi_ok) { napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt"); return NULL; }
	status = get_uint64_from_value(env, args[6], &firstRecovery);
	if(status != napi_ok) { napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt"); return NULL; }

	int numThreads = 0;
	if(argc >= 8) {
		status = napi_get_value_int32(env, args[7], &numThreads);
		if(status != napi_ok) { napi_throw_type_error(env, NULL, "numThreads must be an integer"); return NULL; }
	}

	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	if(inputsLen < (size_t)(numInputs * blockSize)) {
		napi_throw_range_error(env, NULL, "inputs buffer too small for numInputs * blockSize");
		return NULL;
	}

	if(outputsLen < (size_t)(numRecovery * blockSize)) {
		napi_throw_range_error(env, NULL, "outputs buffer too small for numRecovery * blockSize");
		return NULL;
	}

	size_t blockSize64 = (size_t)(blockSize / 8);

	gf64_init_dispatch();

	// AVX-512 alignment gate (matches the other NAPI entry points): the
	// Fenger pipeline's subproduct-tree build dispatches to the AVX-512
	// gf64_poly_mul path, which requires 64-byte-aligned buffers. JS
	// Buffers are only guaranteed base-aligned, so bounce misaligned
	// buffers through aligned temporaries.
	const size_t inputsBytes = (size_t)numInputs * (size_t)blockSize;
	const size_t outputsBytes = (size_t)numRecovery * (size_t)blockSize;
	void* aligned_inputs = (void*)inputs;
	bool inputs_bounced = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)inputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inputsBytes, 64)) {
			memcpy(tmp, inputs, inputsBytes);
			aligned_inputs = tmp;
			inputs_bounced = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed (fenger inputs)");
			return NULL;
		}
	}
	void* aligned_outputs = (void*)outputs;
	bool outputs_bounced = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outputsBytes, 64)) {
			memcpy(tmp, outputs, outputsBytes);
			aligned_outputs = tmp;
			outputs_bounced = true;
		} else {
			if (inputs_bounced) ALIGN_FREE(aligned_inputs);
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed (fenger outputs)");
			return NULL;
		}
	}

	GF64Controller::ComputeRecoveryBlocksFenger(
		(const gf64_t*)aligned_inputs, (size_t)numInputs,
		(gf64_t*)aligned_outputs, (size_t)numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		(int)numThreads
	);

	if (outputs_bounced) {
		memcpy(outputs, aligned_outputs, outputsBytes);
		ALIGN_FREE(aligned_outputs);
	}
	if (inputs_bounced) {
		ALIGN_FREE(aligned_inputs);
	}

	return NULL;
}

// ============================================================================
// v2-4: build_coefficient_matrix NAPI binding
// ----------------------------------------------------------------------------
// Standalone matrix build. Allocates a Buffer of numRecovery × numInputs
// gf64_t, fills it with the Cauchy coefficient matrix, and returns it. The
// JS layer can call this asynchronously (in a worker thread) BEFORE the
// kernel call, to overlap the matrix-build cost with the file-read phase.
// The returned Buffer is opaque from JS's perspective; pass it back to
// compute_recovery_with_coeff to use it.
// ============================================================================
static napi_value BuildCoefficientMatrix_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 4;
	napi_value args[4];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if (status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if (argc < 4) {
		napi_throw_type_error(env, NULL, "Requires numInputs, numRecovery, firstInput, firstRecovery");
		return NULL;
	}

	int64_t numInputs = 0, numRecovery = 0;
	status = napi_get_value_int64(env, args[0], &numInputs);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "numInputs must be an integer"); return NULL; }
	status = napi_get_value_int64(env, args[1], &numRecovery);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "numRecovery must be an integer"); return NULL; }

	uint64_t firstInput = 0, firstRecovery = 0;
	status = get_uint64_from_value(env, args[2], &firstInput);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt"); return NULL; }
	status = get_uint64_from_value(env, args[3], &firstRecovery);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt"); return NULL; }

	if (numInputs <= 0 || numRecovery <= 0) {
		napi_throw_range_error(env, NULL, "numInputs and numRecovery must be positive");
		return NULL;
	}

	gf64_init_dispatch();
	gf64_t* matrix = GF64Controller::BuildCauchyMatrixAlloc(
		(size_t)numInputs, (size_t)numRecovery, firstInput, firstRecovery);
	if (!matrix) {
		napi_throw_error(env, NULL, "matrix allocation failed");
		return NULL;
	}

	// Wrap the C buffer in a NAPI Buffer; free when GC'd.
	napi_value result;
	status = napi_create_buffer_copy(env, (size_t)numRecovery * (size_t)numInputs * sizeof(gf64_t),
	                                  matrix, NULL, &result);
	free(matrix);
	if (status != napi_ok) {
		napi_throw_error(env, NULL, "napi_create_buffer_copy failed");
		return NULL;
	}
	return result;
}

// ============================================================================
// v2-4: compute_recovery_with_coeff NAPI binding
// ----------------------------------------------------------------------------
// Same args as compute_recovery_full, plus a pre-computed coefficient
// matrix Buffer as the first arg (replacing the inputs buffer slot — the
// inputs are passed in arg 5, see below). Layout:
//   coeffMatrix (Buffer), inputs (Buffer), outputs (Buffer),
//   numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]
// Note: firstInput / firstRecovery are accepted for API consistency with
// compute_recovery_full but are NOT used by the kernel path; the matrix
// has been pre-built and the kernel just reads it.
// ============================================================================
static napi_value ComputeRecoveryWithCoeff_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 9;
	napi_value args[9];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if (status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if (argc < 8) {
		napi_throw_type_error(env, NULL, "Requires coeffMatrix, inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	const gf64_t* coeff = NULL;
	size_t coeffLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&coeff, &coeffLen);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "coeffMatrix must be a Buffer"); return NULL; }

	gf64_t* inputs = NULL;
	size_t inputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&inputs, &inputsLen);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "inputs must be a Buffer"); return NULL; }
	void* aligned_inputs = (void*)inputs;
	bool needs_inputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)inputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, inputsLen, 64)) {
			memcpy(tmp, inputs, inputsLen);
			aligned_inputs = tmp;
			needs_inputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[2], (void**)&outputs, &outputsLen);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "outputs must be a Buffer"); return NULL; }
	void* aligned_outputs = (void*)outputs;
	bool needs_outputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		size_t needed = outputsLen;
		if (ALIGN_ALLOC(tmp, needed, 64)) {
			aligned_outputs = tmp;
			needs_outputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int64_t numInputs = 0, numRecovery = 0;
	status = napi_get_value_int64(env, args[3], &numInputs);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "numInputs must be an integer"); return NULL; }
	status = napi_get_value_int64(env, args[4], &numRecovery);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "numRecovery must be an integer"); return NULL; }

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[5], &blockSize);
	if (status != napi_ok) { napi_throw_type_error(env, NULL, "blockSize must be an integer"); return NULL; }

	// args[6] and args[7] (firstInput, firstRecovery) are accepted but unused
	// in the pre-computed matrix path.

	int32_t numThreads = 0;
	if (argc >= 9) {
		status = napi_get_value_int32(env, args[8], &numThreads);
		if (status != napi_ok) { napi_throw_type_error(env, NULL, "numThreads must be an integer"); return NULL; }
	}

	if (numInputs <= 0) { napi_throw_range_error(env, NULL, "numInputs must be positive"); return NULL; }
	if (numRecovery <= 0) { napi_throw_range_error(env, NULL, "numRecovery must be positive"); return NULL; }
	if (blockSize <= 0 || blockSize % 8 != 0) { napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8"); return NULL; }

	size_t blockSize64 = (size_t)(blockSize / 8);
	gf64_init_dispatch();

	GF64Controller::ComputeRecoveryBlocksWithCoeff(
		(gf64_t*)aligned_inputs, (size_t)numInputs,
		(gf64_t*)aligned_outputs, (size_t)numRecovery,
		blockSize64, coeff, (int)numThreads);

	if (needs_inputs_temp) {
		memcpy(inputs, aligned_inputs, (size_t)numInputs * blockSize);
		ALIGN_FREE(aligned_inputs);
	}
	if (needs_outputs_temp) {
		memcpy(outputs, aligned_outputs, (size_t)numRecovery * blockSize);
		ALIGN_FREE(aligned_outputs);
	}

	return NULL;
}

// ============================================================================
// P1 of issue #46's delivery plan: compute_recovery_streaming NAPI binding.
//
// Replaces the per-batch JS recovery loop in lib/par3gen.js's
// _processRecoveryBatch (which makes ~N/BATCH_SIZE JS<->C roundtrips) with a
// single C++ call that accepts ALL the input batches in one go. The JS layer
// can keep the existing per-batch loop as a fallback for builds that lack
// the new entry.
//
// Args:
//   inputBatches   (Array<Buffer>) — one Buffer per batch; each Buffer holds
//                    numBlocksInBatch * blockSize bytes. The buffers are
//                    concatenated in array order to form the full input.
//   outputBuffer   (Buffer)        — numRecovery * blockSize bytes
//   numInputs      (int32)         — total number of input blocks (== sum of
//                    inputBatches[i].length / blockSize)
//   numRecovery    (int32)         — number of recovery blocks to compute
//   blockSize      (int64)         — bytes per block, must be % 8 == 0
//   firstInput     (Number|BigInt) — global index of the first input block
//                    (in the canonical per-batch flow this is 0; the global
//                    indices are contiguous across batches)
//   firstRecovery  (Number|BigInt) — global index of the first recovery block
//   numThreads     (int32, optional) — 0 = auto
//
// Dispatch mirrors lib/par3gen.js's dispatchRecovery so the streaming entry
// produces the same bytes as the per-batch loop on the same workload:
//   1. Fenger  — power-of-2 N, power-of-2 R, blockSize%8==0, non-Windows
//                (default) or env-forced (PAR3_GF64_USE_FENGER=1). Falls
//                back internally to the legacy Cauchy path when constraints
//                aren't met (subproduct-tree requirement).
//   2. Barycentric — N > PAR3_BF64_MIN_INPUTS (default 10000) AND
//                blockSize%8==0.
//   3. Legacy Cauchy — the same engine that compute_recovery (per-batch
//                fallback) calls.
//
// The Cauchy / Barycentric distinction matters: Cauchy is the
// matrix-vector product out[r][w] = XOR_c in[c][w] / (y_r XOR x_c); Barycentric
// is Lagrange interpolation of the same inputs. Both are LINEAR in the
// input blocks, so the streaming entry can take the concatenation of all
// per-batch inputs in one call and produce the same final output as the
// per-batch loop's XOR-accumulation. To preserve bit-exact parity with the
// per-batch path, the streaming entry must pick the same engine the per-batch
// loop would have picked — hence the dispatch mirror here.
// ============================================================================

// defined_win32() — compile-time mirror of JS process.platform === 'win32'
// (mirrors the JS Fenger-gate default in lib/par3gen.js dispatchRecovery).
#if defined(_WIN32)
static inline bool defined_win32(void) { return true; }
#else
static inline bool defined_win32(void) { return false; }
#endif

static napi_value ComputeRecoveryStreaming_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 8;
	napi_value args[8];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 7) {
		napi_throw_type_error(env, NULL,
			"Requires inputBatches (Array<Buffer>), outputBuffer, numInputs, numRecovery, blockSize, firstInput, firstRecovery [, numThreads]");
		return NULL;
	}

	bool is_array = false;
	status = napi_is_array(env, args[0], &is_array);
	if(status != napi_ok || !is_array) {
		napi_throw_type_error(env, NULL, "inputBatches must be an array of Buffers");
		return NULL;
	}
	uint32_t numBatches_u32 = 0;
	status = napi_get_array_length(env, args[0], &numBatches_u32);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get inputBatches array length");
		return NULL;
	}

	gf64_t* outputs = NULL;
	size_t outputsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&outputs, &outputsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "outputBuffer must be a Buffer");
		return NULL;
	}
	void* aligned_outputs = (void*)outputs;
	bool needs_outputs_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)outputs & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, outputsLen, 64)) {
			aligned_outputs = tmp;
			needs_outputs_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed (outputs)");
			return NULL;
		}
	}

	int64_t numInputs = 0;
	status = napi_get_value_int64(env, args[2], &numInputs);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numInputs must be an integer");
		return NULL;
	}

	int64_t numRecovery = 0;
	status = napi_get_value_int64(env, args[3], &numRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numRecovery must be an integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	uint64_t firstInput = 0;
	status = get_uint64_from_value(env, args[5], &firstInput);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt");
		return NULL;
	}

	uint64_t firstRecovery = 0;
	status = get_uint64_from_value(env, args[6], &firstRecovery);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt");
		return NULL;
	}

	int32_t numThreads = 0;
	if(argc >= 8) {
		status = napi_get_value_int32(env, args[7], &numThreads);
		if(status != napi_ok) {
			napi_throw_type_error(env, NULL, "numThreads must be an integer");
			return NULL;
		}
	}

	if(numInputs < 0) {
		napi_throw_range_error(env, NULL, "numInputs must be non-negative");
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}
	if(numRecovery < 0) {
		napi_throw_range_error(env, NULL, "numRecovery must be non-negative");
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}
	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}
	if(numInputs == 0 || numRecovery == 0) {
		// Trivial short-circuit: JS accumulator is already zero-initialised.
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}

	const size_t totalInputsBytes = (size_t)numInputs * (size_t)blockSize;
	if(outputsLen < (size_t)numRecovery * (size_t)blockSize) {
		napi_throw_range_error(env, NULL, "outputBuffer too small for numRecovery * blockSize");
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}

	// 64-byte aligned contiguous input buffer (engine requires contiguity;
	// C++ alloc bypasses JS Buffer kMaxLength, e.g. 10 GiB / 4 KiB = 2.5M blocks).
	gf64_t* inputs = NULL;
	void* aligned_inputs_alloc = nullptr;
	if (ALIGN_ALLOC(aligned_inputs_alloc, totalInputsBytes, 64)) {
		inputs = (gf64_t*)aligned_inputs_alloc;
	} else {
		napi_throw_error(env, NULL, "ALIGN_ALLOC failed (streaming inputs)");
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		return NULL;
	}

	size_t inputsCopiedBlocks = 0;
	for(uint32_t i = 0; i < numBatches_u32; i++) {
		napi_value elem;
		status = napi_get_element(env, args[0], i, &elem);
		if(status != napi_ok) {
			ALIGN_FREE(aligned_inputs_alloc);
			if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
			napi_throw_error(env, NULL, "Failed to read inputBatches element");
			return NULL;
		}
		void* p = NULL;
		size_t pl = 0;
		status = napi_get_buffer_info(env, elem, &p, &pl);
		if(status != napi_ok) {
			ALIGN_FREE(aligned_inputs_alloc);
			if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
			napi_throw_type_error(env, NULL, "inputBatches element must be a Buffer");
			return NULL;
		}
		if(pl % (size_t)blockSize != 0) {
			ALIGN_FREE(aligned_inputs_alloc);
			if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
			napi_throw_range_error(env, NULL, "inputBatches element length not a multiple of blockSize");
			return NULL;
		}
		const size_t blocksInBatch = pl / (size_t)blockSize;
		if(inputsCopiedBlocks + blocksInBatch > (size_t)numInputs) {
			ALIGN_FREE(aligned_inputs_alloc);
			if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
			napi_throw_range_error(env, NULL, "inputBatches total blocks exceeds numInputs");
			return NULL;
		}
		if(pl > 0) {
			memcpy((uint8_t*)inputs + inputsCopiedBlocks * (size_t)blockSize, p, pl);
		}
		inputsCopiedBlocks += blocksInBatch;
	}
	if(inputsCopiedBlocks != (size_t)numInputs) {
		ALIGN_FREE(aligned_inputs_alloc);
		if (needs_outputs_temp) ALIGN_FREE(aligned_outputs);
		napi_throw_range_error(env, NULL, "inputBatches total blocks does not equal numInputs");
		return NULL;
	}

	const size_t blockSize64 = (size_t)(blockSize / 8);

	// Engine dispatch mirrors lib/par3gen.js's dispatchRecovery() so the
	// streaming entry produces bit-exact parity with the per-batch loop.
	// Cauchy / Fenger / Barycentric are all linear in the input blocks,
	// so concatenating all per-batch inputs and calling the engine once
	// is mathematically identical to per-batch calls XORed into an
	// accumulator.
	gf64_init_dispatch();

	const char* fengerEnv = getenv("PAR3_GF64_USE_FENGER");
	const bool fengerForced = fengerEnv && (
		strcmp(fengerEnv, "1") == 0 ||
		strcasecmp(fengerEnv, "true") == 0 ||
		strcasecmp(fengerEnv, "yes") == 0 ||
		strcasecmp(fengerEnv, "on") == 0);
	const bool fengerDisabled = fengerEnv && (
		strcmp(fengerEnv, "0") == 0 ||
		strcasecmp(fengerEnv, "false") == 0 ||
		strcasecmp(fengerEnv, "no") == 0 ||
		strcasecmp(fengerEnv, "off") == 0);
	const bool isPowerOfTwoOrTrivial_N =
		(numInputs == 0) || (numInputs == 1) || ((numInputs > 1) && ((numInputs & (numInputs - 1)) == 0));
	const bool isPowerOfTwoOrTrivial_R =
		(numRecovery == 0) || (numRecovery == 1) || ((numRecovery > 1) && ((numRecovery & (numRecovery - 1)) == 0));
	const bool fengerEligible =
		isPowerOfTwoOrTrivial_N &&
		isPowerOfTwoOrTrivial_R &&
		(blockSize % 8 == 0) &&
		(fengerForced || !defined_win32());

	long baryMin = 10000;
	const char* baryEnv = getenv("PAR3_BF64_MIN_INPUTS");
	if(baryEnv) {
		char* endp = nullptr;
		long v = strtol(baryEnv, &endp, 10);
		if(endp != baryEnv && v > 0) baryMin = v;
	}
	const bool baryEligible = (numInputs > baryMin) && (blockSize % 8 == 0);

	static bool dispatch_warned = false;
	if(fengerEligible && !fengerDisabled) {
		if(!dispatch_warned) {
			dispatch_warned = true;
			std::fprintf(stderr, "[par3gen] compute_recovery_streaming: Fenger kernel for numInputs=%lld numRecovery=%lld\n",
				(long long)numInputs, (long long)numRecovery);
		}
		GF64Controller::ComputeRecoveryBlocksFenger(
			(const gf64_t*)inputs, (size_t)numInputs,
			(gf64_t*)aligned_outputs, (size_t)numRecovery,
			blockSize64,
			firstInput, firstRecovery,
			(int)numThreads
		);
	} else if(baryEligible) {
		if(!dispatch_warned) {
			dispatch_warned = true;
			std::fprintf(stderr, "[par3gen] compute_recovery_streaming: Barycentric kernel for numInputs=%lld\n",
				(long long)numInputs);
		}
		GF64Controller::ComputeRecoveryBlocksBarycentric(
			(const gf64_t*)inputs, (size_t)numInputs,
			(gf64_t*)aligned_outputs, (size_t)numRecovery,
			blockSize64,
			firstInput, firstRecovery,
			(int)numThreads
		);
	} else {
		GF64Controller::ComputeRecoveryBlocks(
			(const gf64_t*)inputs, (size_t)numInputs,
			(gf64_t*)aligned_outputs, (size_t)numRecovery,
			blockSize64,
			firstInput, firstRecovery,
			(int)numThreads
		);
	}

	ALIGN_FREE(aligned_inputs_alloc);
	if (needs_outputs_temp) {
		memcpy(outputs, aligned_outputs, (size_t)numRecovery * (size_t)blockSize);
		ALIGN_FREE(aligned_outputs);
	}

	return NULL;
}

// ComputeRepairBlocks NAPI binding
// Args: availBlocks, repairedBlocks, numAvail, numMissing, blockSize, solveMatrix [, numThreads]
static napi_value ComputeRepair_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 7;
	napi_value args[7];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 6) {
		napi_throw_type_error(env, NULL, "Requires availBlocks, repairedBlocks, numAvail, numMissing, blockSize, solveMatrix [, numThreads]");
		return NULL;
	}

	gf64_t* availBlocks = NULL;
	size_t availLen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&availBlocks, &availLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "availBlocks must be a Buffer");
		return NULL;
	}
	void* aligned_availBlocks = (void*)availBlocks;
	bool needs_availBlocks_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)availBlocks & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, availLen, 64)) {
			memcpy(tmp, availBlocks, availLen);
			aligned_availBlocks = tmp;
			needs_availBlocks_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* repairedBlocks = NULL;
	size_t repairedLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&repairedBlocks, &repairedLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "repairedBlocks must be a Buffer");
		return NULL;
	}
	void* aligned_repairedBlocks = (void*)repairedBlocks;
	bool needs_repairedBlocks_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)repairedBlocks & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, repairedLen, 64)) {
			memcpy(tmp, repairedBlocks, repairedLen);
			aligned_repairedBlocks = tmp;
			needs_repairedBlocks_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t numAvail = 0;
	status = napi_get_value_int32(env, args[2], &numAvail);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numAvail must be an integer");
		return NULL;
	}

	int32_t numMissing = 0;
	status = napi_get_value_int32(env, args[3], &numMissing);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "numMissing must be an integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[4], &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}

	gf64_t* solveMatrix = NULL;
	size_t solveLen = 0;
	status = napi_get_buffer_info(env, args[5], (void**)&solveMatrix, &solveLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "solveMatrix must be a Buffer");
		return NULL;
	}
	void* aligned_solveMatrix = (void*)solveMatrix;
	bool needs_solveMatrix_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)solveMatrix & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, solveLen, 64)) {
			memcpy(tmp, solveMatrix, solveLen);
			aligned_solveMatrix = tmp;
			needs_solveMatrix_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t numThreads = 0;
	if(argc >= 7) {
		status = napi_get_value_int32(env, args[6], &numThreads);
		if(status != napi_ok) {
			numThreads = 0;
		}
	}

	if(numAvail <= 0 || numMissing <= 0) {
		napi_throw_range_error(env, NULL, "numAvail and numMissing must be positive");
		return NULL;
	}

	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	size_t blockSize64 = (size_t)(blockSize / 8);

	if(availLen < (size_t)(numAvail * blockSize)) {
		napi_throw_range_error(env, NULL, "availBlocks buffer too small");
		return NULL;
	}

	if(repairedLen < (size_t)(numMissing * blockSize)) {
		napi_throw_range_error(env, NULL, "repairedBlocks buffer too small");
		return NULL;
	}

	if(solveLen < (size_t)(numMissing * numAvail * sizeof(gf64_t))) {
		napi_throw_range_error(env, NULL, "solveMatrix buffer too small");
		return NULL;
	}

	gf64_init_dispatch();
	GF64Controller::ComputeRepairBlocks(
		(gf64_t*)aligned_availBlocks, (size_t)numAvail,
		(gf64_t*)aligned_repairedBlocks, (size_t)numMissing,
		(gf64_t*)aligned_solveMatrix, blockSize64,
		(int)numThreads
	);

	if (needs_availBlocks_temp) {
		memcpy(availBlocks, aligned_availBlocks, (size_t)numAvail * blockSize);
		ALIGN_FREE(aligned_availBlocks);
	}
	if (needs_repairedBlocks_temp) {
		memcpy(repairedBlocks, aligned_repairedBlocks, (size_t)numMissing * blockSize);
		ALIGN_FREE(aligned_repairedBlocks);
	}
	if (needs_solveMatrix_temp) {
		memcpy(solveMatrix, aligned_solveMatrix, solveLen);
		ALIGN_FREE(aligned_solveMatrix);
	}

	return NULL;
}

static napi_value SolveAndReconstruct_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 4;
	napi_value args[4];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 4) {
		napi_throw_type_error(env, NULL, "Requires A, rhsBlocks, n, blockSize");
		return NULL;
	}

	gf64_t* A = NULL;
	size_t ALen = 0;
	status = napi_get_buffer_info(env, args[0], (void**)&A, &ALen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "A must be a Buffer");
		return NULL;
	}
	void* aligned_A = (void*)A;
	bool needs_A_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)A & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, ALen, 64)) {
			memcpy(tmp, A, ALen);
			aligned_A = tmp;
			needs_A_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	gf64_t* rhsBlocks = NULL;
	size_t rhsLen = 0;
	status = napi_get_buffer_info(env, args[1], (void**)&rhsBlocks, &rhsLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "rhsBlocks must be a Buffer");
		return NULL;
	}
	void* aligned_rhsBlocks = (void*)rhsBlocks;
	bool needs_rhsBlocks_temp = false;
	if (gf64_current_method == GF64_AVX512 && ((uintptr_t)rhsBlocks & 63) != 0) {
		void* tmp = nullptr;
		if (ALIGN_ALLOC(tmp, rhsLen, 64)) {
			memcpy(tmp, rhsBlocks, rhsLen);
			aligned_rhsBlocks = tmp;
			needs_rhsBlocks_temp = true;
		} else {
			napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
			return NULL;
		}
	}

	int32_t n = 0;
	status = napi_get_value_int32(env, args[2], &n);
	if(status != napi_ok || n <= 0) {
		napi_throw_type_error(env, NULL, "n must be a positive integer");
		return NULL;
	}

	int64_t blockSize = 0;
	status = napi_get_value_int64(env, args[3], &blockSize);
	if(status != napi_ok || blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_type_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	size_t nSize = (size_t)n;
	size_t blockSize64 = (size_t)(blockSize / 8);

	if(ALen < nSize * nSize * sizeof(gf64_t)) {
		napi_throw_range_error(env, NULL, "A buffer too small for n×n matrix");
		return NULL;
	}

	if(rhsLen < nSize * (size_t)blockSize) {
		napi_throw_range_error(env, NULL, "rhsBlocks buffer too small for n blocks");
		return NULL;
	}

	gf64_init_dispatch();
	int result = GF64Controller::SolveAndReconstruct((gf64_t*)aligned_A, (gf64_t*)aligned_rhsBlocks, nSize, blockSize64, 0);

	napi_value ret;
	if(result == 0) {
		status = napi_get_boolean(env, true, &ret);
	} else {
		status = napi_get_boolean(env, false, &ret);
	}
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create return value");
		return NULL;
	}

	if (needs_A_temp) {
		memcpy(A, aligned_A, nSize * nSize * sizeof(gf64_t));
		ALIGN_FREE(aligned_A);
	}
	if (needs_rhsBlocks_temp) {
		memcpy(rhsBlocks, aligned_rhsBlocks, nSize * (size_t)blockSize);
		ALIGN_FREE(aligned_rhsBlocks);
	}

	return ret;
}

static void AlignedBufferFinalizer(napi_env env, void* finalize_data, void* hint) {
	(void)env;
	(void)hint;
	ALIGN_FREE(finalize_data);
}

static napi_value AllocAlignedBuffer(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 1;
	napi_value argv[1];

	status = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	int64_t size = 0;
	status = napi_get_value_int64(env, argv[0], &size);
	if(status != napi_ok || size <= 0) {
		napi_throw_range_error(env, NULL, "Size must be positive");
		return NULL;
	}

	void* data = NULL;
	ALIGN_ALLOC(data, (size_t)size, 64);
	if(data == NULL) {
		napi_throw_error(env, NULL, "ALIGN_ALLOC failed");
		return NULL;
	}
	memset(data, 0, (size_t)size);

	napi_value buffer;
	status = napi_create_external_buffer(env, (size_t)size, data, AlignedBufferFinalizer, NULL, &buffer);
	if(status != napi_ok) {
		ALIGN_FREE(data);
		napi_throw_error(env, NULL, "napi_create_external_buffer failed");
		return NULL;
	}
	return buffer;
}

static napi_value IsAlignedBuffer(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 2;
	napi_value argv[2];

	status = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	bool is_buffer = false;
	status = napi_is_buffer(env, argv[0], &is_buffer);
	if(status != napi_ok || !is_buffer) {
		napi_value result;
		napi_get_boolean(env, false, &result);
		return result;
	}

	void* data = NULL;
	size_t length = 0;
	status = napi_get_buffer_info(env, argv[0], &data, &length);
	(void)length;
	if(status != napi_ok || data == NULL) {
		napi_value result;
		napi_get_boolean(env, false, &result);
		return result;
	}

	int64_t alignment = 64;
	if(argc >= 2) {
		status = napi_get_value_int64(env, argv[1], &alignment);
		if(status != napi_ok || alignment <= 0 || (alignment & (alignment - 1)) != 0) {
			/* alignment must be a positive power of two for the bitmask
			 * test to be correct; fall back to 64 on any malformed input */
			alignment = 64;
		}
	}

	bool aligned = (((uintptr_t)data & ((uintptr_t)alignment - 1)) == 0);
	napi_value result;
	napi_get_boolean(env, aligned, &result);
	return result;
}

napi_value parpar_gf64_init_NAPI(napi_env env, napi_value exports) {
	napi_status status;

	napi_property_descriptor properties[] = {
		{ "mul", NULL, Gf64Encoder_NAPI_mul, NULL, NULL, NULL, napi_default, NULL },
		{ "mul_arr", NULL, Gf64Encoder_NAPI_mul_arr, NULL, NULL, NULL, napi_default, NULL },
		{ "coupled_muladd_arr", NULL, Gf64Encoder_NAPI_coupled_muladd_arr, NULL, NULL, NULL, napi_default, NULL },
		{ "fused_output_muladd_arr", NULL, Gf64Encoder_NAPI_fused_output_muladd_arr, NULL, NULL, NULL, napi_default, NULL },
		{ "two_d_muladd_arr", NULL, Gf64Encoder_NAPI_two_d_muladd_arr, NULL, NULL, NULL, napi_default, NULL }
	};

	napi_value constructor;
	status = napi_define_class(env,
		"Gf64Encoder",
		NAPI_AUTO_LENGTH,
		Gf64Encoder_NAPI_constructor,
		NULL,
		5,
		properties,
		&constructor);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to define Gf64Encoder class");
		return NULL;
	}

	status = napi_set_named_property(env, exports, "Gf64Encoder", constructor);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set Gf64Encoder constructor");
		return NULL;
	}

	napi_value gf64_info_fn;
	status = napi_create_function(env, NULL, 0, gf64_info_NAPI, NULL, &gf64_info_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create gf64_info function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "gf64_info", gf64_info_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set gf64_info property");
		return NULL;
	}

napi_value create_fn;
	status = napi_create_function(env, NULL, 0, Gf64Encoder_NAPI_create, NULL, &create_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create Gf64Encoder_create function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "Gf64Encoder_create", create_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set Gf64Encoder_create property");
		return NULL;
	}

	napi_value destroy_fn;
	status = napi_create_function(env, NULL, 0, Gf64Encoder_NAPI_destroy, NULL, &destroy_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create Gf64Encoder_destroy function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "Gf64Encoder_destroy", destroy_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set Gf64Encoder_destroy property");
		return NULL;
	}

	napi_value solve_fn;
	status = napi_create_function(env, NULL, 0, gf64_solve_NAPI, NULL, &solve_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create gf64_solve function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "gf64_solve", solve_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set gf64_solve property");
		return NULL;
	}

	napi_value compute_recovery_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecovery_NAPI, NULL, &compute_recovery_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery", compute_recovery_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery property");
		return NULL;
	}

	napi_value compute_recovery_full_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecoveryFull_NAPI, NULL, &compute_recovery_full_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery_full function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery_full", compute_recovery_full_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery_full property");
		return NULL;
	}

	// T10: barycentric-form recovery variant (subproduct-tree path).
	napi_value compute_recovery_barycentric_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecoveryBarycentric_NAPI, NULL, &compute_recovery_barycentric_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery_barycentric function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery_barycentric", compute_recovery_barycentric_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery_barycentric property");
		return NULL;
	}

	// Issue #28: Fenger-Toeplitz recovery variant. Bit-exact alternative to
	// compute_recovery_full on power-of-2 inputs/recovery counts; falls back
	// to the legacy muladd path otherwise. Opt-in from JS via
	// PAR3_GF64_USE_FENGER=1; default OFF.
	napi_value compute_recovery_fenger_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecoveryFenger_NAPI, NULL, &compute_recovery_fenger_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery_fenger function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery_fenger", compute_recovery_fenger_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery_fenger property");
		return NULL;
	}

	// v2-4: standalone matrix build + kernel-with-precomputed-matrix.
	napi_value build_coeff_fn;
	status = napi_create_function(env, NULL, 0, BuildCoefficientMatrix_NAPI, NULL, &build_coeff_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create build_coefficient_matrix function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "build_coefficient_matrix", build_coeff_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set build_coefficient_matrix property");
		return NULL;
	}

	napi_value compute_with_coeff_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecoveryWithCoeff_NAPI, NULL, &compute_with_coeff_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery_with_coeff function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery_with_coeff", compute_with_coeff_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery_with_coeff property");
		return NULL;
	}

	// P1 of issue #46: streaming variant of compute_recovery_full — accepts
	// multiple batches of input blocks in one call. See ComputeRecoveryStreaming_NAPI
	// for the dispatch mirroring (must match lib/par3gen.js dispatchRecovery).
	napi_value compute_recovery_streaming_fn;
	status = napi_create_function(env, NULL, 0, ComputeRecoveryStreaming_NAPI, NULL, &compute_recovery_streaming_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_recovery_streaming function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_recovery_streaming", compute_recovery_streaming_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_recovery_streaming property");
		return NULL;
	}

	napi_value compute_repair_fn;
	status = napi_create_function(env, NULL, 0, ComputeRepair_NAPI, NULL, &compute_repair_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create compute_repair function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "compute_repair", compute_repair_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set compute_repair property");
		return NULL;
	}

	napi_value solve_reconstruct_fn;
	status = napi_create_function(env, NULL, 0, SolveAndReconstruct_NAPI, NULL, &solve_reconstruct_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create solve_and_reconstruct function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "solve_and_reconstruct", solve_reconstruct_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set solve_and_reconstruct property");
		return NULL;
	}

	napi_value par3_create_streaming_fn;
	status = napi_create_function(env, NULL, 0, par3_create_streaming_NAPI, NULL, &par3_create_streaming_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create par3_create_streaming function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "par3_create_streaming", par3_create_streaming_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set par3_create_streaming property");
		return NULL;
	}

	napi_value alloc_aligned_buffer_fn;
	status = napi_create_function(env, NULL, 0, AllocAlignedBuffer, NULL, &alloc_aligned_buffer_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create allocAlignedBuffer function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "allocAlignedBuffer", alloc_aligned_buffer_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set allocAlignedBuffer property");
		return NULL;
	}

	napi_value is_aligned_buffer_fn;
	status = napi_create_function(env, NULL, 0, IsAlignedBuffer, NULL, &is_aligned_buffer_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to create isAlignedBuffer function");
		return NULL;
	}
	status = napi_set_named_property(env, exports, "isAlignedBuffer", is_aligned_buffer_fn);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to set isAlignedBuffer property");
		return NULL;
	}

	return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, parpar_gf64_init_NAPI)