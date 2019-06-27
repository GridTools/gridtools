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

#ifndef __CUDACC__
#error This is CUDA only header
#endif

#include <type_traits>

#include "../../common/cuda_type_traits.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../dim.hpp"
#include "../esf_metafunctions.hpp"
#include "../execution_types.hpp"
#include "../positional.hpp"
#include "../sid/multi_shift.hpp"
#include "k_cache.hpp"

namespace gridtools {
    namespace cuda {
        enum class sync_type { fill, flush };

        template <sync_type SyncType, class Orig, class Cache>
        GT_FUNCTION_DEVICE std::enable_if_t<SyncType == sync_type::fill> do_sync(
            Orig &&GT_RESTRICT orig, Cache &GT_RESTRICT cache) {
            cache = orig;
        }
        template <sync_type SyncType, class Orig, class Cache>
        GT_FUNCTION_DEVICE std::enable_if_t<SyncType == sync_type::flush> do_sync(
            Orig &GT_RESTRICT orig, Cache &&GT_RESTRICT cache) {
            orig = cache;
        }

        template <class LocalDomain>
        class iterate_domain {
            using caches_t = typename LocalDomain::cache_sequence_t;

            LocalDomain const &m_local_domain;
            typename LocalDomain::ptr_t m_ptr;
            k_caches_holder<typename LocalDomain::ptr_t, typename LocalDomain::strides_t> m_k_caches_holder;

            template <class Arg, class Offset>
            GT_FUNCTION_DEVICE decltype(auto) deref(Offset offset) const {
                auto ptr = device::at_key<Arg>(m_ptr);
                sid::shift(ptr, sid::get_stride<Arg, dim::k>(m_local_domain.m_strides), offset);
                return *ptr;
            }

            template <sync_type SyncType, class Arg, class Offset>
            GT_FUNCTION_DEVICE void sync_at(Offset offset) {
                if (m_local_domain.template validate_k_pos<Arg>(
                        *host_device::at_key<positional<dim::k>>(m_ptr) + offset))
                    do_sync<SyncType>(deref<k_cache_original<Arg>>(offset), deref<Arg>(offset));
            }

            template <class Policy, sync_type SyncType, class Arg>
            GT_FUNCTION_DEVICE void sync_cache(Arg, std::false_type) {
                using stride_t = std::decay_t<decltype(sid::get_stride<Arg, dim::k>(m_local_domain.m_strides))>;
                using sync_point_t = integral_constant<int_t,
                    (execute::is_forward<Policy>::value && SyncType == sync_type::fill) ||
                            (execute::is_backward<Policy>::value && SyncType == sync_type::flush)
                        ? stride_t::plus
                        : stride_t::minus>;
                sync_at<SyncType, Arg>(sync_point_t{});
            }

            template <class Policy, sync_type SyncType, class Arg>
            GT_FUNCTION_DEVICE void sync_cache(Arg, std::true_type) {
                using stride_t = std::decay_t<decltype(sid::get_stride<Arg, dim::k>(m_local_domain.m_strides))>;
                device::for_each<meta::iseq_to_list<std::make_integer_sequence<int_t, stride_t::size>>>([this](auto i) {
                    using sync_point_t = integral_constant<int_t, decltype(i)::value + stride_t::minus>;
                    sync_at<SyncType, Arg>(sync_point_t{});
                });
            }

            template <class Dim, class Ptr, class Offset>
            GT_FUNCTION void shift(Ptr &ptr, Offset offset) {
                sid::shift(ptr, sid::get_stride<Dim>(m_local_domain.m_strides), offset);
            }

          public:
            GT_FUNCTION_DEVICE iterate_domain(LocalDomain const &local_domain, typename LocalDomain::ptr_t ptr)
                : m_local_domain(local_domain), m_ptr(wstd::move(ptr)) {
                bind_k_caches(m_k_caches_holder, m_ptr, m_local_domain.m_strides);
            }

            template <class ExecutionType>
            GT_FUNCTION_DEVICE void increment_k(ExecutionType) {
                shift<dim::k>(m_ptr, k_step<ExecutionType>());
            }

            static constexpr bool has_k_caches = !meta::is_empty<k_caches<caches_t>>::value;

            template <class ExecutionType, class IsFirstLevel>
            GT_FUNCTION_DEVICE void fill_caches(ExecutionType, IsFirstLevel) {
                using filling_cache_args_t = meta::transform<cache_parameter, meta::filter<is_filling_cache, caches_t>>;
                device::for_each<filling_cache_args_t>(
                    [&](auto arg) { sync_cache<ExecutionType, sync_type::fill>(arg, IsFirstLevel{}); });
            }

            template <class ExecutionType, class IsLastLevel>
            GT_FUNCTION_DEVICE void flush_caches(ExecutionType, IsLastLevel) {
                using flushing_cache_args_t =
                    meta::transform<cache_parameter, meta::filter<is_flushing_cache, caches_t>>;
                device::for_each<flushing_cache_args_t>(
                    [&](auto arg) { sync_cache<ExecutionType, sync_type::flush>(arg, IsLastLevel{}); });
            }

            GT_FUNCTION decltype(auto) ptr() const { return m_ptr; }
            GT_FUNCTION decltype(auto) strides() const { return m_local_domain.m_strides; }
        };
    } // namespace cuda
} // namespace gridtools
