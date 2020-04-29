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

#include <type_traits>
#include <utility>

#include "../common/array.hpp"
#include "../common/cuda_runtime.hpp"
#include "../common/cuda_util.hpp"
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/utility.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "info.hpp"

namespace gridtools {
    namespace storage {
        namespace gpu_impl_ {
            /**
             * @brief metafunction used to retrieve a layout_map with n-dimensions that can be used in combination with
             * the GPU backend (i-first order). E.g., make_layout<5> will return following type: layout_map<4,3,2,1,0>.
             * This means the i-dimension (value: 4) is coalesced in memory, followed by the j-dimension (value: 3),
             * followed by the k-dimension (value: 2), followed by the fourth dimension (value: 1), etc. The reason for
             * having i as innermost is because of the gridtools execution model. The GPU backend will give best
             * performance (in most cases) when using the provided layout.
             */
            template <size_t N, class = std::make_index_sequence<N>>
            struct make_layout;

            template <size_t N, size_t... Dims>
            struct make_layout<N, std::index_sequence<Dims...>> {
                using type = layout_map<(N - 1 - Dims)...>;
            };

            template <class T, size_t N>
            struct target_view {
                T *m_ptr;
                storage::info<N> m_info;

#if defined(GT_CUDA_ARCH) or (defined(GT_CUDACC) and defined(__clang__))
                GT_FUNCTION_DEVICE auto const &info() const { return m_info; }

                GT_FUNCTION_DEVICE auto *data() const { return m_ptr; }

                template <class... Args>
                GT_FUNCTION_DEVICE auto operator()(Args &&... args) const
                    -> decltype(m_ptr[m_info.index(wstd::forward<Args>(args)...)]) {
                    return m_ptr[m_info.index(wstd::forward<Args>(args)...)];
                }

                GT_FUNCTION_DEVICE decltype(auto) operator()(array<int, N> const &arg) const {
                    return m_ptr[m_info.index(arg)];
                }

                GT_FUNCTION_DEVICE GT_CONSTEXPR auto length() const { return m_info.length(); }

                GT_FUNCTION_DEVICE GT_CONSTEXPR auto const &lengths() const { return m_info.lengths(); }
#endif
            };
        } // namespace gpu_impl_

        struct gpu {
            friend std::false_type storage_is_host_referenceable(gpu) { return {}; }

            template <size_t Dims>
            friend typename gpu_impl_::make_layout<Dims>::type storage_layout(
                gpu, std::integral_constant<size_t, Dims>) {
                return {};
            }

            friend integral_constant<size_t, 128> storage_alignment(gpu) { return {}; }

            template <class LazyType, class T = typename LazyType::type>
            friend auto storage_allocate(gpu, LazyType, size_t size) {
                return cuda_util::cuda_malloc<T[]>(size);
            }

            template <class T>
            friend void storage_update_target(gpu, T *dst, T const *src, size_t size) {
                GT_CUDA_CHECK(cudaMemcpy(const_cast<std::remove_volatile_t<T> *>(dst),
                    const_cast<std::remove_volatile_t<T> *>(src),
                    size * sizeof(T),
                    cudaMemcpyHostToDevice));
            }

            template <class T>
            friend void storage_update_host(gpu, T *dst, T const *src, size_t size) {
                GT_CUDA_CHECK(cudaMemcpy(const_cast<std::remove_volatile_t<T> *>(dst),
                    const_cast<std::remove_volatile_t<T> *>(src),
                    size * sizeof(T),
                    cudaMemcpyDeviceToHost));
            }

            template <class T, size_t N>
            friend gpu_impl_::target_view<T, N> storage_make_target_view(gpu, T *ptr, info<N> const &info) {
                return {ptr, info};
            }
        };
    } // namespace storage
} // namespace gridtools
