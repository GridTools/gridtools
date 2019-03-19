/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include "../GCL.hpp"

#ifdef GCL_HOSTWORKAROUND

#include "../../common/cuda_util.hpp"

namespace gridtools {
    namespace _impl {
        enum alloc_type { host_normal, host_page_locked };
        template <typename T, alloc_type>
        struct helper_alloc {};

        // manage page locked memory on the host
        template <typename T>
        struct helper_alloc<T, host_page_locked> {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr;
                    GT_CUDA_CHECK(cudaMallocHost(&ptr, sz * sizeof(T)));
                    return ptr;
                } else {
                    return nullptr;
                }
            }

            static void free(T *t) { GT_CUDA_CHECK(cudaFreeHost(t)); }

            static T *realloc(T *t, size_t sz) {
                free(t);
                return alloc(sz);
            }
        };

        // manage normal memory on the host
        template <typename T>
        struct helper_alloc<T, host_normal> {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr = malloc(sz);
                    return ptr;
                } else {
                    return 0;
                }
            }

            static void free(T *t) { free(t); }

            static T *realloc(T *t, size_t sz) {
                free(t);
                return alloc(sz);
            }
        };
    } // namespace _impl
} // namespace gridtools
#endif
