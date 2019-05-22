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
#include <utility>

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/at_key.hpp>

#include "../../../common/array.hpp"
#include "../../../common/cuda_type_traits.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../backend_cuda/iterate_domain_cache.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../dim.hpp"
#include "../../esf_metafunctions.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../../pos3.hpp"
#include "../../positional.hpp"
#include "../../sid/multi_shift.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <class IterateDomainArguments>
    class iterate_domain_cuda {
        GT_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using caches_t = typename IterateDomainArguments::local_domain_t::cache_sequence_t;
        using ij_cache_args_t = ij_cache_args<caches_t>;
        using k_cache_args_t = k_cache_args<caches_t>;
        using readwrite_args_t = compute_readwrite_args<typename IterateDomainArguments::esf_sequence_t>;
        using cache_sequence_t = typename IterateDomainArguments::local_domain_t::cache_sequence_t;
        using iterate_domain_cache_t = iterate_domain_cache<IterateDomainArguments>;

      public:
        using shared_iterate_domain_t = typename iterate_domain_cache_t::ij_caches_tuple_t;

      private:
        local_domain_t const &m_local_domain;
        typename local_domain_t::ptr_t m_ptr;
        int_t m_block_size_i;
        int_t m_block_size_j;
        shared_iterate_domain_t *GT_RESTRICT m_pshared_iterate_domain;
        iterate_domain_cache_t m_iterate_domain_cache;
        int_t m_thread_pos[2];

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

        template <class Arg, class Accessor>
        GT_FUNCTION auto get_ptr(Accessor const &acc) const {
            auto ptr = host_device::at_key<Arg>(m_ptr);
            sid::multi_shift<Arg>(ptr, m_local_domain.m_strides, acc);
            return ptr;
        }

      public:
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_k(Offset offset = {}) {
            sid::shift(m_ptr, sid::get_stride<dim::k>(m_local_domain.m_strides), offset);
        }

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<int_t> begin, pos3<int_t> block_no, pos3<int_t> pos_in_block) {
            host_device::for_each_type<typename local_domain_t::esf_args_t>(
                initialize_index<typename IterateDomainArguments::backend_t, local_domain_t>(
                    m_local_domain.m_strides, begin, block_no, pos_in_block, m_ptr));
        }

        GT_FUNCTION int_t i() const { return pos<dim::i>(); }

        GT_FUNCTION int_t j() const { return pos<dim::j>(); }

        GT_FUNCTION int_t k() const { return pos<dim::k>(); }

      public:
        static constexpr bool has_ij_caches = !meta::is_empty<ij_caches<cache_sequence_t>>::value;
        static constexpr bool has_k_caches = !meta::is_empty<k_caches<cache_sequence_t>>::value;

        GT_FUNCTION_DEVICE iterate_domain_cuda(
            local_domain_t const &local_domain, int_t block_size_i, int_t block_size_j)
            : m_local_domain(local_domain), m_ptr(local_domain.m_ptr_holder()), m_block_size_i(block_size_i),
              m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <class Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            return Extent::iminus::value <= m_thread_pos[0] &&
                   Extent::iplus::value > m_thread_pos[0] - m_block_size_i &&
                   Extent::jminus::value <= m_thread_pos[1] && Extent::jplus::value > m_thread_pos[1] - m_block_size_j;
        }

        GT_FUNCTION_DEVICE void set_block_pos(int_t ipos, int_t jpos) {
            m_thread_pos[0] = ipos;
            m_thread_pos[1] = jpos;
        }

        GT_FUNCTION_DEVICE void set_shared_iterate_domain_pointer(shared_iterate_domain_t *ptr) {
            m_pshared_iterate_domain = ptr;
        }

        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GT_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template slide_caches<IterationPolicy>();
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         */
        template <typename IterationPolicy>
        GT_FUNCTION void fill_caches(bool first_level) {
            GT_STATIC_ASSERT(is_iteration_policy<IterationPolicy>::value, GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template fill_caches<IterationPolicy>(*this, first_level);
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         */
        template <typename IterationPolicy>
        GT_FUNCTION void flush_caches(bool last_level) {
            GT_STATIC_ASSERT(is_iteration_policy<IterationPolicy>::value, GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template flush_caches<IterationPolicy>(*this, last_level);
        }

        template <class Arg, class DataStore = typename Arg::data_store_t, class Data = typename DataStore::data_t>
        GT_FUNCTION Data *deref_for_k_cache(int_t k_offset) const {
            auto k_pos = k() + k_offset;
            if (k_pos < 0 ||
                k_pos >= host_device::at_key<typename DataStore::storage_info_t>(m_local_domain.m_ksize_map))
                return nullptr;
            Data *res = host_device::at_key<Arg>(this->m_ptr);
            sid::shift(res, sid::get_stride<Arg, dim::k>(m_local_domain.m_strides), k_offset);
            return res;
        }

        template <class Arg, class Accessor, std::enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
            return boost::fusion::at_key<Arg>(*m_pshared_iterate_domain).at(m_thread_pos[0], m_thread_pos[1], acc);
        }

        template <class Arg,
            class Accessor,
            std::enable_if_t<meta::st_contains<k_cache_args_t, Arg>::value &&
                                 !meta::st_contains<ij_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
            return m_iterate_domain_cache.template get_k_cache<Arg>(acc);
        }

        template <class Arg,
            class Accessor,
            std::enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value &&
                                 !meta::st_contains<k_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION decltype(auto) deref(Accessor const &acc) const {
            return dereference<Arg>(this->template get_ptr<Arg>(acc));
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
