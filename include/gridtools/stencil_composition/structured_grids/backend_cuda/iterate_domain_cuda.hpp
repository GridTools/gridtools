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
#include "../../../common/host_device.hpp"
#include "../../../meta.hpp"
#include "../../backend_cuda/iterate_domain_cache.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../esf_metafunctions.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <class IterateDomainArguments>
    class iterate_domain_cuda : public iterate_domain<IterateDomainArguments> {
        GT_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        using base_t = iterate_domain<IterateDomainArguments>;

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using caches_t = typename IterateDomainArguments::local_domain_t::cache_sequence_t;
        using ij_cache_args_t = GT_META_CALL(ij_cache_args, caches_t);
        using k_cache_args_t = GT_META_CALL(k_cache_args, caches_t);

        using readwrite_args_t = GT_META_CALL(compute_readwrite_args, typename IterateDomainArguments::esf_sequence_t);

        // array storing the (i,j) position of the current thread within the block
        array<int, 2> m_thread_pos;

        using cache_sequence_t = typename IterateDomainArguments::local_domain_t::cache_sequence_t;

      public:
        using iterate_domain_cache_t = iterate_domain_cache<IterateDomainArguments>;

        typedef typename iterate_domain_cache_t::ij_caches_tuple_t shared_iterate_domain_t;

      private:
        using base_t::increment_i;
        using base_t::increment_j;

        const uint_t m_block_size_i;
        const uint_t m_block_size_j;
        shared_iterate_domain_t *GT_RESTRICT m_pshared_iterate_domain;
        iterate_domain_cache_t m_iterate_domain_cache;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        template <class Arg, class T>
        static GT_FUNCTION enable_if_t<!meta::st_contains<readwrite_args_t, Arg>::value && is_texture_type<T>::value, T>
        dereference(T *ptr) {
            return __ldg(ptr);
        }
#endif
        template <class Arg, class Ptr>
        static GT_FUNCTION auto dereference(Ptr ptr) GT_AUTO_RETURN(*ptr);

      public:
        static constexpr bool has_ij_caches = !meta::is_empty<GT_META_CALL(ij_caches, cache_sequence_t)>::value;
        static constexpr bool has_k_caches = !meta::is_empty<GT_META_CALL(k_caches, cache_sequence_t)>::value;

        template <class LocalDomain>
        GT_FUNCTION_DEVICE iterate_domain_cuda(LocalDomain &&local_domain, uint_t block_size_i, uint_t block_size_j)
            : base_t(wstd::forward<LocalDomain>(local_domain)), m_block_size_i(block_size_i),
              m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            return m_thread_pos[0] >= Extent::iminus::value &&
                   m_thread_pos[0] < (int)m_block_size_i + Extent::iplus::value &&
                   m_thread_pos[1] >= Extent::jminus::value &&
                   m_thread_pos[1] < (int)m_block_size_j + Extent::jplus::value;
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
            auto offset = host_device::at_key<Arg>(this->m_index);
            sid::shift(offset, sid::get_stride<Arg, dim::k>(this->m_local_domain.m_strides), k_offset);

            return offset < host_device::at_key<typename DataStore::storage_info_t>(
                                this->m_local_domain.m_total_length_map) &&
                           offset >= 0
                       ? host_device::at_key<Arg>(this->m_ptr) + offset
                       : nullptr;
        }

        template <class Arg, class Accessor, enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
            return boost::fusion::at_key<Arg>(*m_pshared_iterate_domain).at(m_thread_pos[0], m_thread_pos[1], acc);
        }

        template <class Arg,
            class Accessor,
            enable_if_t<meta::st_contains<k_cache_args_t, Arg>::value &&
                            !meta::st_contains<ij_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
            return m_iterate_domain_cache.template get_k_cache<Arg>(acc);
        }

        template <class Arg,
            class Accessor,
            enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value &&
                            !meta::st_contains<k_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION auto deref(Accessor const &acc) const
            GT_AUTO_RETURN(dereference<Arg>(this->template get_ptr<Arg>(acc)));
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
