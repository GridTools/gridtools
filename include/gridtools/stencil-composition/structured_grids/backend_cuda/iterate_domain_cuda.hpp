/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#ifndef __CUDACC__
#error This is CUDA only header
#endif

#include <type_traits>

#include <boost/type_traits/is_arithmetic.hpp>

#include "../../../common/cuda_type_traits.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../esf_metafunctions.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cuda
        : public iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> // CRTP
    {
        iterate_domain_cuda(iterate_domain_cuda const &) = delete;
        iterate_domain_cuda &operator=(iterate_domain_cuda const &) = delete;

        GRIDTOOLS_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        typedef iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> super;
        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename local_domain_t::esf_args local_domain_args_t;

        using readwrite_args_t = typename compute_readwrite_args<typename IterateDomainArguments::esf_sequence_t>::type;

        // array storing the (i,j) position of the current thread within the block
        array<int, 2> m_thread_pos;

      public:
        typedef typename super::strides_cached_t strides_cached_t;

        typedef typename super::iterate_domain_cache_t iterate_domain_cache_t;

        typedef shared_iterate_domain<strides_cached_t,
            typename IterateDomainArguments::max_extent_t,
            typename iterate_domain_cache_t::ij_caches_tuple_t>
            shared_iterate_domain_t;

        static constexpr bool has_ij_caches = iterate_domain_cache_t::has_ij_caches;

      private:
        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::k_caches_map_t k_caches_map_t;

        using super::increment_i;
        using super::increment_j;

        const uint_t m_block_size_i;
        const uint_t m_block_size_j;
        shared_iterate_domain_t *RESTRICT m_pshared_iterate_domain;
        iterate_domain_cache_t m_iterate_domain_cache;

      public:
        __forceinline__ __device__ iterate_domain_cuda(
            local_domain_t const &local_domain, uint_t block_size_i, uint_t block_size_j)
            : super(local_domain), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        __forceinline__ __device__ bool is_thread_in_domain() const {
            return m_thread_pos[0] >= Extent::iminus::value &&
                   m_thread_pos[0] < ((int)m_block_size_i + Extent::iplus::value) &&
                   m_thread_pos[1] >= Extent::jminus::value &&
                   m_thread_pos[1] < ((int)m_block_size_j + Extent::jplus::value);
        }

        __forceinline__ __device__ void set_block_pos(int_t ipos, int_t jpos) {
            m_thread_pos[0] = ipos;
            m_thread_pos[1] = jpos;
        }

        GT_FUNCTION void set_shared_iterate_domain_pointer_impl(shared_iterate_domain_t *ptr) {
            m_pshared_iterate_domain = ptr;
        }
        GT_FUNCTION strides_cached_t const &RESTRICT strides_impl() const {
            return m_pshared_iterate_domain->strides();
        }
        GT_FUNCTION strides_cached_t &RESTRICT strides_impl() { return m_pshared_iterate_domain->strides(); }

        /**
         * @brief metafunction that determines if an accessor is accessed via kcache register set
         */
        template <typename Accessor>
        struct accessor_from_kcache_reg : index_is_cached<Accessor::index_t::value, k_caches_map_t> {
            GRIDTOOLS_STATIC_ASSERT(is_accessor<Accessor>::value, GT_INTERNAL_ERROR);
        };

        /** @brief return a value that was cached
         * specialization where cache goes via shared memory
         */
        template <size_t Index,
            class ReturnType,
            class Accessor,
            enable_if_t<index_is_cached<Index, ij_caches_map_t>::value, int> = 0>
        GT_FUNCTION ReturnType get_cache_value_impl(Accessor const &acc) const {
            // retrieve the ij cache from the fusion tuple and access the element required give the current thread
            // position within
            // the block and the offsets of the accessor
            return m_pshared_iterate_domain->template get_ij_cache<static_uint<Index>>().at<0>(m_thread_pos, acc);
        }

        /** @brief return a value that was cached
         * specialization where cache goes via kcache register set
         *
         */
        template <size_t Index,
            class ReturnType,
            class Accessor,
            enable_if_t<!index_is_cached<Index, ij_caches_map_t>::value, int> = 0>
        GT_FUNCTION ReturnType get_cache_value_impl(Accessor const &acc) const {
            return m_iterate_domain_cache.template get_k_cache<static_uint<Index>>().at(acc);
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        template <class Arg, class T>
        static GT_FUNCTION
            enable_if_t<!boost::mpl::has_key<readwrite_args_t, Arg>::value && is_texture_type<T>::value, T>
            deref_impl(T const *ptr) {
            return __ldg(ptr);
        }
#endif

        template <class Arg, class Ptr>
        static GT_FUNCTION auto deref_impl(Ptr ptr) GT_AUTO_RETURN(*ptr);

        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template slide_caches<IterationPolicy>();
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy>
        GT_FUNCTION void fill_caches(bool first_level) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template fill_caches<IterationPolicy>(*this, first_level);
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy>
        GT_FUNCTION void flush_caches(bool last_level) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template flush_caches<IterationPolicy>(*this, last_level);
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
