/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "../GCL.hpp"

#ifdef HOSTWORKAROUND
namespace gridtools {
    namespace _impl {
        enum alloc_type { host_normal, host_page_locked };
        template < typename T, alloc_type >
        struct helper_alloc {};

        // manage page locked memory on the host
        template < typename T >
        struct helper_alloc< T, host_page_locked > {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr;
                    cudaError_t status = cudaMallocHost(&ptr, sz * sizeof(T));
                    if (!checkCudaStatus(status)) {
                        printf("Allocation did not succed\n");
                        exit(1);
                    }
                    return ptr;
                } else {
                    return NULL;
                }
            }

            static void free(T *t) {
                if (!t)
                    cudaError_t status = cudaFreeHost(t);
            }

            static T *realloc(T *t, size_t sz) {
                if (t != 0) {
                    free(t);
                }
                return alloc(sz);
            }
        };

        // manage normal memory on the host
        template < typename T >
        struct helper_alloc< T, host_normal > {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr = malloc(sz);
                    return ptr;
                } else {
                    return 0;
                }
            }

            static void free(T *t) {
                if (!t)
                    free(t);
            }

            static T *realloc(T *t, size_t sz) {
                if (!t) {
                    free(t);
                }
                return alloc(sz);
            }
        };
    } // namespace _impl
} // namespace gridtools
#endif
