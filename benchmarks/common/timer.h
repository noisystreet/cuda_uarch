// SPDX-License-Identifier: MIT
#pragma once

#include "config.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string_view>
#include <vector>

namespace uarch
{

    // ─── Host-side wall-clock timer ────────────────────────────────────────────
    class HostTimer
    {
    public:
        void start() noexcept { start_ = Clock::now(); }
        void stop() noexcept { stop_ = Clock::now(); }

        /// Elapsed in milliseconds
        [[nodiscard]] double elapsed_ms() const noexcept
        {
            return std::chrono::duration<double, std::milli>(stop_ - start_).count();
        }

        /// Elapsed in microseconds
        [[nodiscard]] double elapsed_us() const noexcept
        {
            return std::chrono::duration<double, std::micro>(stop_ - start_).count();
        }

    private:
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point start_;
        Clock::time_point stop_;
    };

    // ─── GPU-side timer using CUDA events ─────────────────────────────────────
    class GpuTimer
    {
    public:
        GpuTimer()
        {
            CUDA_CHECK(cudaEventCreate(&start_));
            CUDA_CHECK(cudaEventCreate(&stop_));
        }

        ~GpuTimer()
        {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        // Non-copyable, movable
        GpuTimer(const GpuTimer &) = delete;
        GpuTimer &operator=(const GpuTimer &) = delete;
        GpuTimer(GpuTimer &&other) noexcept : start_(other.start_), stop_(other.stop_)
        {
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }

        void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

        void stop(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }

        /// Synchronize and return elapsed time in milliseconds
        [[nodiscard]] double elapsed_ms()
        {
            CUDA_CHECK(cudaEventSynchronize(stop_));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
            return static_cast<double>(ms);
        }

        /// Synchronize and return elapsed time in microseconds
        [[nodiscard]] double elapsed_us() { return elapsed_ms() * 1000.0; }

    private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    };

    // ─── Statistics collector ──────────────────────────────────────────────────
    struct SampleStats
    {
        double mean_us;
        double median_us;
        double min_us;
        double max_us;
        double stddev_us;
    };

    /// Compute statistics from a vector of samples (microseconds assumed)
    inline SampleStats compute_stats(const std::vector<double> &samples_us)
    {
        SampleStats stats{};
        if (samples_us.empty())
            return stats;

        // Sort for median
        auto sorted = samples_us;
        std::sort(sorted.begin(), sorted.end());

        stats.min_us = sorted.front();
        stats.max_us = sorted.back();
        stats.median_us = sorted[sorted.size() / 2];

        double sum = 0.0;
        for (auto v : sorted)
            sum += v;
        stats.mean_us = sum / sorted.size();

        double sq_sum = 0.0;
        for (auto v : sorted)
        {
            double diff = v - stats.mean_us;
            sq_sum += diff * diff;
        }
        stats.stddev_us = std::sqrt(sq_sum / sorted.size());

        return stats;
    }

    inline void print_stats(std::string_view label, const std::vector<double> &samples_us)
    {
        auto s = compute_stats(samples_us);
        printf("%-50s  median=%8.2f us  mean=%8.2f us  "
               "min=%8.2f  max=%8.2f  std=%6.2f\n",
               label.data(), s.median_us, s.mean_us, s.min_us, s.max_us, s.stddev_us);
    }

    /// Print stats and format as CSV line
    inline void print_csv(std::string_view label, const std::vector<double> &samples_us)
    {
        auto s = compute_stats(samples_us);
        printf("RESULT,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", label.data(), s.median_us, s.mean_us,
               s.min_us, s.max_us, s.stddev_us, samples_us.size());
    }

} // namespace uarch
