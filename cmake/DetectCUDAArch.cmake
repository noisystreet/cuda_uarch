# DetectCUDAArch.cmake
# Auto-detect local GPU compute capability; fallback to sm_80 (Ampere)
#
# Usage: detect_cuda_arch()
# Sets:  CMAKE_CUDA_ARCHITECTURES

function(detect_cuda_arch)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(STATUS "CUDA architectures already set: ${CMAKE_CUDA_ARCHITECTURES}")
        return()
    endif()

    # Try deviceQuery which ships with CUDA Toolkit
    find_program(DEVICE_QUERY NAMES deviceQuery
        PATHS /usr/local/cuda/samples /opt/cuda/samples
        PATH_SUFFIXES 1_Utilities/deviceQuery
        NO_DEFAULT_PATH
    )

    if(DEVICE_QUERY)
        execute_process(
            COMMAND ${DEVICE_QUERY}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE DEVICE_QUERY_OUT
            TIMEOUT 10
            ERROR_QUIET
        )
        if(DEVICE_QUERY_OUT MATCHES "CUDA Capability Major/Minor version number:[ \t]*([0-9]+)\\.([0-9]+)")
            set(MAJOR ${CMAKE_MATCH_1})
            set(MINOR ${CMAKE_MATCH_2})
            set(CMAKE_CUDA_ARCHITECTURES "${MAJOR}${MINOR}" PARENT_SCOPE)
            message(STATUS "Detected local GPU compute capability: sm_${MAJOR}${MINOR}")
            return()
        endif()
    endif()

    # Fallback: try nvidia-smi
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE CC_STR
            TIMEOUT 5
            ERROR_QUIET
        )
        if(CC_STR MATCHES "^([0-9]+)\\.([0-9]+)$")
            set(MAJOR ${CMAKE_MATCH_1})
            set(MINOR ${CMAKE_MATCH_2})
            set(CMAKE_CUDA_ARCHITECTURES "${MAJOR}${MINOR}" PARENT_SCOPE)
            message(STATUS "Detected GPU compute capability via nvidia-smi: sm_${MAJOR}${MINOR}")
            return()
        endif()
    endif()

    # Final fallback
    set(CMAKE_CUDA_ARCHITECTURES "80" PARENT_SCOPE)
    message(WARNING "Could not detect GPU compute capability. Falling back to sm_80 (Ampere).")
endfunction()
