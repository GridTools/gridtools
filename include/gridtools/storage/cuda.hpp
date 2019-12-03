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
#include "../common/cuda_util.hpp"
#include "../common/generic_metafunctions/utility.hpp"
#include "../common/hip_wrappers.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "info.hpp"

namespace gridtools {
    namespace storage {
        namespace cuda_impl_ {
            /**
             * @brief metafunction used to retrieve a layout_map with n-dimensions
             * that can be used in combination with the GPU backend (i-first order).
             * E.g., make_layout<5> will return following type: layout_map<4, 3, 2, 1, 0>.
             * This means the i-dimension (value: 4) is coalesced in memory, followed
             * by the j-dimension (value: 3), followed by the k-dimension (value: 2), followed
             * by the fourth dimension (value: 1), etc. The reason for having i as innermost
             * is because of the gridtools execution model. The GPU backend will give best
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
                storage::info<N> const *m_info;

                using data_t = T;

                GT_FUNCTION_DEVICE auto const &info() const { return *m_info; }

                GT_FUNCTION_DEVICE auto *data() const { return m_ptr; }

                template <class... Args>
                GT_FUNCTION_DEVICE auto operator()(Args &&... args) const
                    -> decltype(m_ptr[m_info->index(wstd::forward<Args>(args)...)]) {
                    return m_ptr[m_info->index(wstd::forward<Args>(args)...)];
                }

                GT_FUNCTION_DEVICE decltype(auto) operator()(array<int, N> const &arg) const {
                    return m_ptr[m_info->index(arg)];
                }

                GT_FUNCTION_DEVICE GT_CONSTEXPR auto length() const { return m_info->length(); }

                GT_FUNCTION_DEVICE GT_CONSTEXPR auto const &lengths() const { return m_info->lengths(); }
            };

            template <size_t N>
            auto make_cache(info<N> const &info) {
                return std::make_pair(info, cuda_util::make_clone(info).release());
            }

            template <class Kind, size_t N>
            auto *get_info_ptr(Kind, info<N> const &src) {
                thread_local static auto cache = make_cache(src);
                if (cache.first != src)
                    cache = make_cache(src);
                return cache.second;
            }
        } // namespace cuda_impl_

        struct cuda {
            friend std::false_type storage_is_host_referenceable(cuda) { return {}; }

            template <size_t Dims>
            friend typename cuda_impl_::make_layout<Dims>::type storage_layout(
                cuda, std::integral_constant<size_t, Dims>) {
                return {};
            }

#ifdef __HIPCC__
            friend integral_constant<size_t, 16> storage_alignment(cuda) { return {}; }
#else
            friend integral_constant<size_t, 32> storage_alignment(cuda) { return {}; }
#endif

            template <class LazyType, class T = typename LazyType::type>
            friend auto storage_allocate(cuda, LazyType, size_t size) {
                return cuda_util::cuda_malloc<T[]>(size);
            }

            template <class T>
            friend void storage_update_target(cuda, T *dst, T const *src, size_t size) {
                GT_CUDA_CHECK(cudaMemcpy(const_cast<std::remove_volatile_t<T> *>(dst),
                    const_cast<std::remove_volatile_t<T> *>(src),
                    size * sizeof(T),
                    cudaMemcpyHostToDevice));
            }

            template <class T>
            friend void storage_update_host(cuda, T *dst, T const *src, size_t size) {
                GT_CUDA_CHECK(cudaMemcpy(const_cast<std::remove_volatile_t<T> *>(dst),
                    const_cast<std::remove_volatile_t<T> *>(src),
                    size * sizeof(T),
                    cudaMemcpyDeviceToHost));
            }

            template <class T, class Kind, size_t N>
            friend cuda_impl_::target_view<T, N> storage_make_target_view(
                cuda, Kind kind, T *ptr, info<N> const &info) {
                return {ptr, cuda_impl_::get_info_ptr(kind, info)};
            }
        };
    } // namespace storage
} // namespace gridtools
