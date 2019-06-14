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
#include "../block.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../dim.hpp"
#include "../esf_metafunctions.hpp"
#include "../execution_types.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../positional.hpp"
#include "../run_functor_arguments.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools {
    namespace cuda {
        template <class LocalDomain, class EsfSequence>
        class iterate_domain {
            using caches_t = typename LocalDomain::cache_sequence_t;
            using k_cache_args_t = k_cache_args<caches_t>;
            using readwrite_args_t = compute_readwrite_args<EsfSequence>;
            using k_cache_map_t = get_k_cache_storage_map<caches_t, EsfSequence>;

            LocalDomain const &m_local_domain;
            typename LocalDomain::ptr_t m_ptr;
            int_t m_block_size_i;
            int_t m_block_size_j;
            int_t m_thread_pos[2];
            k_cache_map_t m_k_cache_map;

            template <class Args, class Policy, sync_type SyncType>
            GT_FUNCTION_DEVICE void sync_caches(bool sync_all) {
                host_device::for_each<Args>([&](auto arg) {
                    host_device::at_key<decltype(arg)>(m_k_cache_map).template sync<Policy, SyncType>(*this, sync_all);
                });
            }

            template <class Dim>
            GT_FUNCTION int_t pos() const {
                return *host_device::at_key<positional<Dim>>(m_ptr);
            }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            template <class Arg, class T>
            static GT_FUNCTION
                std::enable_if_t<!meta::st_contains<readwrite_args_t, Arg>::value && is_texture_type<T>::value, T>
                dereference(T *ptr) {
                return __ldg(ptr);
            }
#endif
            template <class Arg, class Ptr>
            static GT_FUNCTION decltype(auto) dereference(Ptr ptr) {
                return *ptr;
            }

            template <class Dim, class Ptr, class Offset>
            GT_FUNCTION void shift(Ptr &ptr, Offset offset) {
                sid::shift(ptr, sid::get_stride<Dim>(m_local_domain.m_strides), offset);
            }

          public:
            GT_FUNCTION_DEVICE iterate_domain(LocalDomain const &local_domain,
                int_t block_size_i,
                int_t block_size_j,
                int_t ipos,
                int_t jpos,
                int_t kpos)
                : m_local_domain(local_domain), m_ptr(local_domain.m_ptr_holder()), m_block_size_i(block_size_i),
                  m_block_size_j(block_size_j), m_thread_pos{ipos, jpos} {
                typename LocalDomain::ptr_diff_t ptr_offset{};
                shift<sid::blocked_dim<dim::i>>(ptr_offset, blockIdx.x);
                shift<sid::blocked_dim<dim::j>>(ptr_offset, blockIdx.y);
                shift<dim::i>(ptr_offset, ipos);
                shift<dim::j>(ptr_offset, jpos);
                shift<dim::k>(ptr_offset, kpos);
                m_ptr = m_ptr + ptr_offset;
            }

            template <class Offset = integral_constant<int_t, 1>>
            GT_FUNCTION void increment_k(Offset offset = {}) {
                shift<dim::k>(m_ptr, offset);
            }

            template <class Offset = integral_constant<int_t, 1>>
            GT_FUNCTION void increment_c(Offset offset = {}) {
                shift<dim::c>(m_ptr, offset);
            }

            GT_FUNCTION int_t i() const { return pos<dim::i>(); }

            GT_FUNCTION int_t j() const { return pos<dim::j>(); }

            GT_FUNCTION int_t k() const { return pos<dim::k>(); }

            GT_FUNCTION_DEVICE static negation<meta::is_empty<ij_caches<caches_t>>> has_ij_caches() { return {}; }
            static constexpr bool has_k_caches = !meta::is_empty<k_caches<caches_t>>::value;

            /**
             * @brief determines whether the current (i,j) position is within the block size
             */
            template <class Extent>
            GT_FUNCTION_DEVICE bool is_thread_in_domain() const {
                return Extent::iminus::value <= m_thread_pos[0] &&
                       Extent::iplus::value > m_thread_pos[0] - m_block_size_i &&
                       Extent::jminus::value <= m_thread_pos[1] &&
                       Extent::jplus::value > m_thread_pos[1] - m_block_size_j;
            }

            GT_FUNCTION_DEVICE void set_block_pos(int_t ipos, int_t jpos) {
                m_thread_pos[0] = ipos;
                m_thread_pos[1] = jpos;
            }

            template <class ExecutionType>
            GT_FUNCTION_DEVICE void slide_caches() {
                device::for_each<k_cache_args_t>(
                    [&](auto arg) { device::at_key<decltype(arg)>(m_k_cache_map).template slide<ExecutionType>(); });
            }

            /**
             * fill next k level from main memory for all k caches. The position of the kcache being filled
             * depends on the iteration policy
             * \tparam IterationPolicy forward: backward
             */
            template <class ExecutionType>
            GT_FUNCTION_DEVICE void fill_caches(bool first_level) {
                using filling_cache_args_t = meta::transform<cache_parameter, meta::filter<is_filling_cache, caches_t>>;
                sync_caches<filling_cache_args_t, ExecutionType, sync_type::fill>(first_level);
            }

            /**
             * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
             * depends on the iteration policy
             * \tparam IterationPolicy forward: backward
             */
            template <class ExecutionType>
            GT_FUNCTION_DEVICE void flush_caches(bool last_level) {
                using flushing_cache_args_t =
                    meta::transform<cache_parameter, meta::filter<is_flushing_cache, caches_t>>;
                sync_caches<flushing_cache_args_t, ExecutionType, sync_type::flush>(last_level);
            }

            template <class Arg, class DataStore = typename Arg::data_store_t, class Data = typename DataStore::data_t>
            GT_FUNCTION_DEVICE Data *deref_for_k_cache(int_t k_offset) const {
                if (!m_local_domain.template validate_k_pos<Arg>(k() + k_offset))
                    return nullptr;
                Data *res = device::at_key<Arg>(m_ptr);
                sid::shift(res, sid::get_stride<Arg, dim::k>(m_local_domain.m_strides), k_offset);
                return res;
            }

            template <class Arg,
                class Accessor,
                std::enable_if_t<meta::st_contains<k_cache_args_t, Arg>::value, int> = 0>
            GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
                return host_device::at_key<Arg>(const_cast<k_cache_map_t &>(m_k_cache_map)).at(acc);
            }

            template <class Arg,
                class Accessor,
                std::enable_if_t<!meta::st_contains<k_cache_args_t, Arg>::value, int> = 0>
            GT_FUNCTION decltype(auto) deref(Accessor const &acc) const {
                auto ptr = host_device::at_key<Arg>(m_ptr);
                sid::multi_shift<Arg>(ptr, m_local_domain.m_strides, acc);
                return dereference<Arg>(ptr);
            }
        };
    } // namespace cuda

    template <class LocalDomain, class EsfSequence>
    struct is_iterate_domain<cuda::iterate_domain<LocalDomain, EsfSequence>> : std::true_type {};
} // namespace gridtools
