// par3_create_streaming NAPI binding — async fd/path-based zero-copy PAR3 create.
//
// Exposed as `gf64.par3_create_streaming(sourcePath, options, callback)`.
//
// Simplified signature (vs the plan's fd/offset/length variant):
//   - The JS layer in B1 can map an fs.open handle onto this path-based
//     signature; the full fd/offset/length API is deferred to a follow-up.
//   - Internally: opens the source via open(2) + fstat(2), mmaps when
//     PAR3_GF64_USE_MMAP=1, otherwise reads the file into a 64-byte-aligned
//     buffer, then runs GF64Controller::ComputeRecoveryBlocksFull on the
//     mapped/buffered pointer.
//   - ASYNC (B1 step 3 overlap): the read + kernel run on the libuv worker
//     pool (napi_async_work), so the JS read/hash pipeline can overlap the
//     recovery computation instead of blocking the event loop.
//   - Returns { recoveryBytes, throughputMBps, durationMs, recoveryBuffer }
//     via callback. recoveryBuffer is an external Buffer (GC-finalized);
//     on NAPI failure the recovery is freed and the legacy path stays
//     authoritative.

#include <node_api.h>

#if defined(_WIN32)
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <malloc.h>     // _aligned_malloc on MSVC
#  define PARPAR_STREAMING_USE_MMAP 0
#  define open  _open
#  define close _close
// MSVC's _fstati64 expects `struct _stati64`, not `struct stat`. We use the
// _stati64 type throughout this file on Windows to keep the call sites
// source-portable. _fstati64 returns the same fields.
#  define stat _stati64
#  define fstat _fstati64
#  define lseek _lseek
#  define O_RDONLY _O_RDONLY
#  define O_BINARY _O_BINARY
#  define read  _read
#  define ssize_t int
// _aligned_malloc is in global namespace on MSVC, NOT std::.
#  define aligned_alloc(alignment, size) _aligned_malloc((size_t)(size), (size_t)(alignment))
#  define aligned_free _aligned_free
#else
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  define PARPAR_STREAMING_USE_MMAP 1
#  define aligned_free(ptr) std::free(ptr)
#endif

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>

#include "gf64_global.h"
#include "par3_engine.h"

/* A2-rev: the recovery buffer outlives the NAPI call — freed by the GC
 * finalizer on the external buffer (must match the aligned_alloc used at
 * the call site). */
static void Par3csRecoveryFinalizer(napi_env env, void* data, void* hint) {
	(void)env; (void)hint;
	if(data) aligned_free(data);
}

/* Async work context: inputs parsed on the main thread, the read+kernel in
 * the execute callback, the result + callback in the complete callback. */
struct Par3csWork {
	char sourcePath[4096];
	int32_t numRecovery;
	int64_t blockSize;
	uint64_t firstInput;
	uint64_t firstRecovery;
	int32_t numThreads;
	napi_async_work work;
	napi_ref callbackRef;
	/* outputs (written by execute, read by complete) */
	bool ok;
	char errorCode[32];
	char errorMsg[512];
	uint8_t* recovery;
	size_t recoveryBytes;
	double durationMs;
	double throughputMBps;
};

static void Par3csExecute(napi_env env, void* data) {
	(void)env;
	Par3csWork* ctx = (Par3csWork*)data;
	ctx->ok = false;
	ctx->recovery = NULL;
	ctx->recoveryBytes = 0;
	ctx->errorCode[0] = '\0';
	ctx->errorMsg[0] = '\0';

	int fd = ::open(ctx->sourcePath, O_RDONLY
#ifndef O_BINARY
	               /* POSIX: binary is the default */
#else
	               | O_BINARY
#endif
	);
	if(fd < 0) {
		int err = errno;
		const char* nodeCode = "EIO";
		switch(err) {
			case ENOENT:  nodeCode = "ENOENT";  break;
			case EACCES:  nodeCode = "EACCES";  break;
			case EISDIR:  nodeCode = "EISDIR";  break;
			case EFBIG:   nodeCode = "EFBIG";   break;
			case ENOMEM:  nodeCode = "ENOMEM";  break;
			case ENAMETOOLONG: nodeCode = "ENAMETOOLONG"; break;
			default: break;
		}
		std::snprintf(ctx->errorCode, sizeof(ctx->errorCode), "%s", nodeCode);
		std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "open failed for sourcePath: errno=%d (%s)", err, std::strerror(err));
		return;
	}

	struct stat st;
	if(::fstat(fd, &st) != 0) {
		int err = errno;
		::close(fd);
		std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "fstat failed: errno=%d (%s)", err, std::strerror(err));
		return;
	}

	if(st.st_size <= 0) {
		::close(fd);
		std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "sourcePath is empty");
		return;
	}

	size_t totalBytes = (size_t)st.st_size;
	size_t numInputs = (size_t)((totalBytes + (size_t)ctx->blockSize - 1) / (size_t)ctx->blockSize);
	size_t blockSize64 = (size_t)(ctx->blockSize / 8);
	uint8_t* mappedPtr = NULL;
	bool mmapActive = false;

#if PARPAR_STREAMING_USE_MMAP
	const char* useMmapEnv = std::getenv("PAR3_GF64_USE_MMAP");
	if(useMmapEnv != NULL && useMmapEnv[0] == '1') {
		void* mm = ::mmap(NULL, totalBytes, PROT_READ, MAP_PRIVATE, fd, 0);
		if(mm != MAP_FAILED) {
			mappedPtr = (uint8_t*)mm;
			mmapActive = true;
		}
	}
#endif
	uint8_t* readBuffer = NULL;
	if(mappedPtr == NULL) {
		readBuffer = (uint8_t*)aligned_alloc(64, (totalBytes + 63) & ~((size_t)63));
		if(readBuffer == NULL) {
			::close(fd);
			std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "aligned_alloc failed for source buffer");
			return;
		}
		size_t off = 0;
		while(off < totalBytes) {
			ssize_t n = ::read(fd, readBuffer + off, totalBytes - off);
			if(n < 0) {
				if(errno == EINTR) continue;
				int err = errno;
				aligned_free(readBuffer);
				::close(fd);
				std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "read failed: errno=%d (%s)", err, std::strerror(err));
				return;
			}
			if(n == 0) {
				aligned_free(readBuffer);
				::close(fd);
				std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "unexpected EOF while reading sourcePath");
				return;
			}
			off += (size_t)n;
		}
		mappedPtr = readBuffer;
	}

	const uint8_t* inputsBytes = mappedPtr;
	size_t recoveryBytes = (size_t)ctx->numRecovery * (size_t)ctx->blockSize;
	uint8_t* recovery = (uint8_t*)aligned_alloc(64, (recoveryBytes + 63) & ~((size_t)63));
	if(recovery == NULL) {
#if PARPAR_STREAMING_USE_MMAP
		if(mmapActive) ::munmap((void*)mappedPtr, totalBytes);
#endif
		if(readBuffer) aligned_free(readBuffer);
		::close(fd);
		std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "aligned_alloc failed for recovery buffer");
		return;
	}
	std::memset(recovery, 0, recoveryBytes);

	gf64_init_dispatch();
	auto t0 = std::chrono::steady_clock::now();
	GF64Controller::ComputeRecoveryBlocksFull(
		(const gf64_t*)inputsBytes, numInputs,
		(gf64_t*)recovery, (size_t)ctx->numRecovery,
		blockSize64,
		ctx->firstInput, ctx->firstRecovery,
		(int)ctx->numThreads
	);
	auto t1 = std::chrono::steady_clock::now();

	double durationMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
	double throughputMBps = 0.0;
	if(durationMs > 0.0) {
		throughputMBps = ((double)recoveryBytes / 1048576.0) / (durationMs / 1000.0);
	}

#if PARPAR_STREAMING_USE_MMAP
	if(mmapActive) {
		::munmap((void*)mappedPtr, totalBytes);
	}
#endif
	if(readBuffer) {
		aligned_free(readBuffer);
	}
	::close(fd);

	ctx->ok = true;
	ctx->recovery = recovery;
	ctx->recoveryBytes = recoveryBytes;
	ctx->durationMs = durationMs;
	ctx->throughputMBps = throughputMBps;
}

static void Par3csComplete(napi_env env, napi_status status, void* data) {
	Par3csWork* ctx = (Par3csWork*)data;
	napi_status s;

	if(status != napi_ok) {
		ctx->ok = false;
		if(ctx->errorMsg[0] == '\0') {
			std::snprintf(ctx->errorMsg, sizeof(ctx->errorMsg), "async work failed (napi status %d)", (int)status);
		}
	}

	napi_value cbArgs[2];
	if(!ctx->ok) {
		/* error: cb(err, NULL) */
		napi_value errVal;
		napi_value msgVal;
		napi_create_string_utf8(env, ctx->errorMsg[0] ? ctx->errorMsg : "par3_create_streaming failed", NAPI_AUTO_LENGTH, &msgVal);
		s = napi_create_error(env, NULL, msgVal, &errVal);
		if(s != napi_ok) {
			napi_get_undefined(env, &errVal);
		}
		cbArgs[0] = errVal;
		napi_get_null(env, &cbArgs[1]);
		if(ctx->recovery) {
			/* execute failed after the recovery alloc — free it here (the
			 * external buffer was never created). */
			aligned_free(ctx->recovery);
			ctx->recovery = NULL;
		}
	} else {
		/* success: cb(null, { recoveryBytes, throughputMBps, durationMs, recoveryBuffer }) */
		napi_value resultObj;
		s = napi_create_object(env, &resultObj);
		if(s != napi_ok) {
			aligned_free(ctx->recovery);
			ctx->recovery = NULL;
			napi_value errVal;
			napi_value msgVal;
			napi_create_string_utf8(env, "Failed to create result object", NAPI_AUTO_LENGTH, &msgVal);
			napi_create_error(env, NULL, msgVal, &errVal);
			cbArgs[0] = errVal;
			napi_get_null(env, &cbArgs[1]);
		} else {
			napi_value recoveryBytesVal;
			napi_create_uint32(env, (uint32_t)(ctx->recoveryBytes & 0xFFFFFFFF), &recoveryBytesVal);
			napi_set_named_property(env, resultObj, "recoveryBytes", recoveryBytesVal);

			napi_value throughputVal;
			napi_create_double(env, ctx->throughputMBps, &throughputVal);
			napi_set_named_property(env, resultObj, "throughputMBps", throughputVal);

			napi_value durationVal;
			napi_create_double(env, ctx->durationMs, &durationVal);
			napi_set_named_property(env, resultObj, "durationMs", durationVal);

			/* A2-rev: hand the recovery buffer to JS (external buffer,
			 * GC-finalized). Ownership (and freeing) belongs to the external
			 * buffer from here on — on failure, free it and the legacy path
			 * remains authoritative. */
			napi_value recoveryBufferVal;
			s = napi_create_external_buffer(env, ctx->recoveryBytes, ctx->recovery, Par3csRecoveryFinalizer, NULL, &recoveryBufferVal);
			if(s != napi_ok) {
				aligned_free(ctx->recovery);
			} else {
				napi_set_named_property(env, resultObj, "recoveryBuffer", recoveryBufferVal);
			}
			ctx->recovery = NULL; /* owned by the external buffer (or freed) */

			napi_get_null(env, &cbArgs[0]);
			cbArgs[1] = resultObj;
		}
	}

	napi_value callback;
	if(napi_get_reference_value(env, ctx->callbackRef, &callback) == napi_ok) {
		napi_value cbReturn;
		napi_call_function(env, callback, callback, 2, cbArgs, &cbReturn);
	}
	napi_delete_reference(env, ctx->callbackRef);
	napi_delete_async_work(env, ctx->work);
	std::free(ctx);
}

static napi_status par3cs_get_uint64(napi_env env, napi_value val, uint64_t* result);

napi_value par3_create_streaming_NAPI(napi_env env, napi_callback_info info) {
	napi_status status;
	size_t argc = 3;
	napi_value args[3];

	status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
	if(status != napi_ok) {
		napi_throw_error(env, NULL, "Failed to get callback info");
		return NULL;
	}

	if(argc < 3) {
		napi_throw_type_error(env, NULL, "par3_create_streaming requires (sourcePath, options, callback)");
		return NULL;
	}

	char sourcePath[4096];
	size_t pathLen = 0;
	status = napi_get_value_string_utf8(env, args[0], sourcePath, sizeof(sourcePath), &pathLen);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "sourcePath must be a string");
		return NULL;
	}
	if(pathLen == 0) {
		napi_throw_type_error(env, NULL, "sourcePath must be a non-empty string");
		return NULL;
	}
	sourcePath[sizeof(sourcePath) - 1] = '\0';

	napi_valuetype cbType;
	status = napi_typeof(env, args[2], &cbType);
	if(status != napi_ok || cbType != napi_function) {
		napi_throw_type_error(env, NULL, "callback must be a function");
		return NULL;
	}

	napi_valuetype optsType;
	status = napi_typeof(env, args[1], &optsType);
	if(status != napi_ok || optsType != napi_object) {
		napi_throw_type_error(env, NULL, "options must be an object");
		return NULL;
	}

	napi_value recoverySlicesVal;
	int32_t numRecovery = 0;
	status = napi_get_named_property(env, args[1], "recoverySlices", &recoverySlicesVal);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "options.recoverySlices is required");
		return NULL;
	}
	status = napi_get_value_int32(env, recoverySlicesVal, &numRecovery);
	if(status != napi_ok || numRecovery <= 0) {
		napi_throw_range_error(env, NULL, "recoverySlices must be a positive integer");
		return NULL;
	}

	napi_value blockSizeVal;
	int64_t blockSize = 0;
	status = napi_get_named_property(env, args[1], "blockSize", &blockSizeVal);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "options.blockSize is required");
		return NULL;
	}
	status = napi_get_value_int64(env, blockSizeVal, &blockSize);
	if(status != napi_ok) {
		napi_throw_type_error(env, NULL, "blockSize must be an integer");
		return NULL;
	}
	if(blockSize <= 0 || blockSize % 8 != 0) {
		napi_throw_range_error(env, NULL, "blockSize must be positive and a multiple of 8");
		return NULL;
	}

	uint64_t firstInput = 0;
	napi_value firstInputVal;
	if(napi_get_named_property(env, args[1], "firstInput", &firstInputVal) == napi_ok) {
		napi_valuetype ft;
		if(napi_typeof(env, firstInputVal, &ft) == napi_ok && ft != napi_undefined && ft != napi_null) {
			if(par3cs_get_uint64(env, firstInputVal, &firstInput) != napi_ok) {
				napi_throw_type_error(env, NULL, "firstInput must be a Number or BigInt");
				return NULL;
			}
		}
	}

	uint64_t firstRecovery = 0;
	napi_value firstRecoveryVal;
	if(napi_get_named_property(env, args[1], "firstRecovery", &firstRecoveryVal) == napi_ok) {
		napi_valuetype ft;
		if(napi_typeof(env, firstRecoveryVal, &ft) == napi_ok && ft != napi_undefined && ft != napi_null) {
			if(par3cs_get_uint64(env, firstRecoveryVal, &firstRecovery) != napi_ok) {
				napi_throw_type_error(env, NULL, "firstRecovery must be a Number or BigInt");
				return NULL;
			}
		}
	}

	int32_t numThreads = 0;
	napi_value numThreadsVal;
	if(napi_get_named_property(env, args[1], "numThreads", &numThreadsVal) == napi_ok) {
		napi_valuetype nt;
		if(napi_typeof(env, numThreadsVal, &nt) == napi_ok && nt != napi_undefined && nt != napi_null) {
			if(napi_get_value_int32(env, numThreadsVal, &numThreads) != napi_ok) {
				napi_throw_type_error(env, NULL, "numThreads must be an integer");
				return NULL;
			}
		}
	}

	Par3csWork* ctx = (Par3csWork*)std::malloc(sizeof(Par3csWork));
	if(ctx == NULL) {
		napi_throw_error(env, NULL, "malloc failed for par3_create_streaming context");
		return NULL;
	}
	std::memset(ctx, 0, sizeof(*ctx));
	std::snprintf(ctx->sourcePath, sizeof(ctx->sourcePath), "%s", sourcePath);
	ctx->numRecovery = numRecovery;
	ctx->blockSize = blockSize;
	ctx->firstInput = firstInput;
	ctx->firstRecovery = firstRecovery;
	ctx->numThreads = numThreads;

	if(napi_create_reference(env, args[2], 1, &ctx->callbackRef) != napi_ok) {
		std::free(ctx);
		napi_throw_error(env, NULL, "Failed to create callback reference");
		return NULL;
	}

	napi_value workName;
	napi_create_string_utf8(env, "par3_create_streaming", NAPI_AUTO_LENGTH, &workName);
	if(napi_create_async_work(env, NULL, workName, Par3csExecute, Par3csComplete, ctx, &ctx->work) != napi_ok) {
		napi_delete_reference(env, ctx->callbackRef);
		std::free(ctx);
		napi_throw_error(env, NULL, "Failed to create async work");
		return NULL;
	}

	if(napi_queue_async_work(env, ctx->work) != napi_ok) {
		napi_delete_reference(env, ctx->callbackRef);
		napi_delete_async_work(env, ctx->work);
		std::free(ctx);
		napi_throw_error(env, NULL, "Failed to queue async work");
		return NULL;
	}

	return NULL; /* undefined — the result arrives via the callback */
}

static napi_status par3cs_get_uint64(napi_env env, napi_value val, uint64_t* result) {
	napi_status status;
	napi_valuetype valuetype;

	status = napi_typeof(env, val, &valuetype);
	if(status != napi_ok) return status;

	if(valuetype == napi_bigint) {
		bool lossless = false;
		status = napi_get_value_bigint_uint64(env, val, result, &lossless);
		if(status == napi_ok) return status;
	}

	int64_t tmp;
	status = napi_get_value_int64(env, val, &tmp);
	if(status == napi_ok) {
		*result = (uint64_t)tmp;
		return napi_ok;
	}

	return napi_generic_failure;
}
