// SPDX-License-Identifier: MIT
#pragma once

#include "config.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>

namespace uarch
{

    // ─── Device memory RAII wrapper ────────────────────────────────────────────
    template <typename T> class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(size_t count) : count_(count)
        {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }

        ~DeviceBuffer()
        {
            if (ptr_)
                cudaFree(ptr_);
        }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), count_(other.count_)
        {
            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        [[nodiscard]] T *get() noexcept { return ptr_; }
        [[nodiscard]] const T *get() const noexcept { return ptr_; }
        [[nodiscard]] size_t size() const noexcept { return count_; }
        [[nodiscard]] size_t bytes() const noexcept { return count_ * sizeof(T); }

        /// Upload from host
        void upload(const T *host, cudaStream_t stream = nullptr)
        {
            CUDA_CHECK(cudaMemcpyAsync(ptr_, host, bytes(), cudaMemcpyHostToDevice, stream));
        }

        /// Download to host
        void download(T *host, cudaStream_t stream = nullptr)
        {
            CUDA_CHECK(cudaMemcpyAsync(host, ptr_, bytes(), cudaMemcpyDeviceToHost, stream));
        }

        /// Fill with a constant value on device
        void fill(T val, cudaStream_t stream = nullptr)
        {
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes(), stream));
            // For non-zero fill, launch a simple kernel would be needed.
            // For zero, cudaMemset works.
            if constexpr (std::is_same_v<T, float>)
            {
                (void)val;
                // zero is the most common fill for benchmarks
                CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes(), stream));
            }
        }

    private:
        T *ptr_ = nullptr;
        size_t count_ = 0;
    };

    // ─── Host-side data generator ──────────────────────────────────────────────
    template <typename T> inline void fill_random(T *data, size_t n, unsigned seed = 42)
    {
        std::mt19937 rng(seed);
        if constexpr (std::is_same_v<T, float>)
        {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < n; ++i)
                data[i] = dist(rng);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < n; ++i)
                data[i] = dist(rng);
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            std::uniform_int_distribution<int> dist(-100, 100);
            for (size_t i = 0; i < n; ++i)
                data[i] = dist(rng);
        }
        else
        {
            // default: bytes
            std::uniform_int_distribution<int> dist(0, 255);
            for (size_t i = 0; i < n; ++i)
                reinterpret_cast<uint8_t *>(data)[i] = static_cast<uint8_t>(dist(rng));
        }
    }

    /// Warm-up: run one or more empty kernel launches to avoid cold-start bias.
    inline void warm_up_device(int iterations = 3)
    {
        for (int i = 0; i < iterations; ++i)
        {
            CUDA_CHECK(cudaFree(nullptr)); // driver init
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

} // namespace uarch
