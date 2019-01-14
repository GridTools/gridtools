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

#include <type_traits>

#include "../../../common/cuda_type_traits.hpp"
#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../esf_metafunctions.hpp"
#include "../grid_traits.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cuda
        : public iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);

        typedef iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> super;
        typedef typename IterateDomainArguments::local_domain_t local_domain_t;

        using super::increment_i;
        using super::increment_j;

        using readwrite_args_t = typename compute_readwrite_args<typename IterateDomainArguments::esf_sequence_t>::type;

      public:
        typedef typename super::strides_cached_t strides_cached_t;
        typedef typename super::iterate_domain_cache_t iterate_domain_cache_t;

        typedef shared_iterate_domain<strides_cached_t,
            typename IterateDomainArguments::max_extent_t,
            typename iterate_domain_cache_t::ij_caches_tuple_t>
            shared_iterate_domain_t;

      private:
        shared_iterate_domain_t *RESTRICT m_pshared_iterate_domain;
        uint_t m_block_size_i;
        uint_t m_block_size_j;
        // array storing the (i,j) position of the current thread within the block
        array<int, 2> m_thread_pos;

      public:
        static constexpr bool has_ij_caches = iterate_domain_cache_t::has_ij_caches;

        GT_FUNCTION iterate_domain_cuda(local_domain_t const &local_domain, uint_t block_size_i, uint_t block_size_j)
            : super(local_domain), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
            return m_thread_pos[0] >= Extent::iminus::value &&
                   m_thread_pos[0] < (int)m_block_size_i + Extent::iplus::value &&
                   m_thread_pos[1] >= Extent::jminus::value &&
                   m_thread_pos[1] < (int)m_block_size_j + Extent::jplus::value;
        }

        GT_FUNCTION
        void set_block_pos(int_t ipos, int_t jpos) {
            m_thread_pos[0] = ipos;
            m_thread_pos[1] = jpos;
        }

        GT_FUNCTION
        void set_shared_iterate_domain_pointer_impl(shared_iterate_domain_t *ptr) { m_pshared_iterate_domain = ptr; }

        GT_FUNCTION
        strides_cached_t const &RESTRICT strides_impl() const {
            //        assert((m_pshared_iterate_domain);
            return m_pshared_iterate_domain->strides();
        }
        GT_FUNCTION
        strides_cached_t &RESTRICT strides_impl() {
            //        assert((m_pshared_iterate_domain));
            return m_pshared_iterate_domain->strides();
        }

        /** @brief return a value that was cached
         */
        template <size_t Index, uint_t Color, typename ReturnType, typename Accessor>
        GT_FUNCTION ReturnType get_cache_value_impl(Accessor const &_accessor) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            // retrieve the ij cache from the fusion tuple and access the element required give the current thread
            // position within the block and the offsets of the accessor
            return m_pshared_iterate_domain->template get_ij_cache<static_uint<Index>>().template at<Color>(
                m_thread_pos, _accessor);
        }

        /** @brief return a the value in memory pointed to by an accessor
         * specialization where the accessor points to an arg which is readonly for all the ESFs in all MSSs
         * Value is read via texture system
         */
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

        // kcaches not yet implemented
        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }
        template <typename IterationPolicy>
        GT_FUNCTION void flush_caches(bool) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }
        template <typename IterationPolicy>
        GT_FUNCTION void fill_caches(bool) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
