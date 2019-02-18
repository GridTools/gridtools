/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#ifdef GT_USE_GPU

#include <cstdlib>

#include <cuda_runtime.h>

#include "../../common/cuda_util.hpp"

namespace _impl {
    inline int get_local_rank() {
        for (auto var : {"MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"})
            if (auto *str = std::getenv(var))
                return std::atoi(str);
        return 0;
    }

    inline int dev_device_count() {
        if (auto *str = std::getenv("NUM_GPU_DEVICES"))
            return std::atoi(str);
        int res;
        GT_CUDA_CHECK(cudaGetDeviceCount(&res));
        return res;
    }
} // namespace _impl

inline void device_binding() { GT_CUDA_CHECK(cudaSetDevice(_impl::get_local_rank() % _impl::dev_device_count())); }

#else

inline void device_binding() {}

#endif
