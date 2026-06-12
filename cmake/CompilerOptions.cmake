# CompilerOptions.cmake
# Sets consistent NVCC and host compiler flags

function(set_cuda_flags)
    # ── Common CUDA flags ───────────────────────────────────────────────
    set(flags
        --expt-extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        --restrict
        --Wno-deprecated-gpu-targets
    )

    # ── Device debug info (always keep line info for profiling) ─────────
    list(APPEND flags -lineinfo)

    # ── Optimisation ────────────────────────────────────────────────────
    list(APPEND flags -O3)
    list(APPEND flags --ftz=true --prec-div=false --prec-sqrt=false)

    # ── Verbose ──────────────────────────────────────────────────────────
    if(BUILD_VERBOSE)
        list(APPEND flags --verbose)
    endif()

    string(REPLACE ";" " " CMAKE_CUDA_FLAGS_STR "${flags}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_STR}" PARENT_SCOPE)

    # ── Host flags ──────────────────────────────────────────────────────
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native" PARENT_SCOPE)

    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS_STR}")
endfunction()
