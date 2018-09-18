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

#include <boost/type_traits/is_arithmetic.hpp>

#include "../../../common/cuda_type_traits.hpp"
#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cuda
        : public iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);

        typedef iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> super;
        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename local_domain_t::esf_args local_domain_args_t;

      public:
        /**
         * metafunction that computes the return type of all operator() of an accessor.
         *
         * If the template argument is not an accessor `type` is mpl::void_
         *
         */
        template <typename Accessor>
        struct accessor_return_type {
            typedef typename super::template accessor_return_type<Accessor>::type type;
        };

        typedef typename super::strides_cached_t strides_cached_t;

        typedef typename super::iterate_domain_cache_t iterate_domain_cache_t;
        typedef typename super::readonly_args_indices_t readonly_args_indices_t;

        typedef shared_iterate_domain<strides_cached_t,
            typename IterateDomainArguments::max_extent_t,
            typename iterate_domain_cache_t::ij_caches_tuple_t>
            shared_iterate_domain_t;

        static constexpr bool has_ij_caches = iterate_domain_cache_t::has_ij_caches;

      private:
        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::k_caches_map_t k_caches_map_t;
        typedef typename iterate_domain_cache_t::bypass_caches_set_t bypass_caches_set_t;
        typedef typename super::reduction_type_t reduction_type_t;

        using super::get_value;
        using super::increment_i;
        using super::increment_j;

        const uint_t m_block_size_i;
        const uint_t m_block_size_j;
        shared_iterate_domain_t *RESTRICT m_pshared_iterate_domain;
        iterate_domain_cache_t m_iterate_domain_cache;

      public:
        GT_FUNCTION
        explicit iterate_domain_cuda(local_domain_t const &local_domain,
            const reduction_type_t &reduction_initial_value,
            const uint_t block_size_i,
            const uint_t block_size_j)
            : super(local_domain, reduction_initial_value), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {
        }

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            return (m_thread_pos[0] >= Extent::iminus::value &&
                    m_thread_pos[0] < ((int)m_block_size_i + Extent::iplus::value) &&
                    m_thread_pos[1] >= Extent::jminus::value &&
                    m_thread_pos[1] < ((int)m_block_size_j + Extent::jplus::value));
        }

        GT_FUNCTION
        void set_block_pos(const int_t ipos, const int_t jpos) {
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

        /** @brief metafunction that determines if an arg is pointing to a field which is read only by all ESFs
         */
        template <typename Accessor>
        struct accessor_points_to_readonly_arg {

            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);

            typedef typename boost::mpl::at<local_domain_args_t,
                boost::mpl::integral_c<int, Accessor::index_t::value>>::type arg_t;

            typedef typename boost::mpl::has_key<readonly_args_indices_t,
                boost::mpl::integral_c<int, arg_index<arg_t>::value>>::type type;
        };

        /**
         * @brief metafunction that determines if an accessor has to be read from texture memory
         */
        template <typename Accessor>
        struct accessor_read_from_texture {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_<typename accessor_points_to_readonly_arg<Accessor>::type,
                    typename boost::mpl::not_<typename boost::mpl::has_key<bypass_caches_set_t,
                        static_uint<Accessor::index_t::value>>::type                          // mpl::has_key
                        >::type                                                               // mpl::not,
                    >::type,                                                                  // mpl::(inner)and_
                typename is_texture_type<typename accessor_return_type<Accessor>::type>::type // is_texture_type
                >::type type;
        };

        /**
         * @brief metafunction that determines if an accessor is accessed via shared memory
         */
        template <typename Accessor>
        struct accessor_from_shared_mem {
            typedef typename boost::remove_reference<Accessor>::type acc_t;

            GRIDTOOLS_STATIC_ASSERT((is_accessor<acc_t>::value), GT_INTERNAL_ERROR);
            typedef static_uint<acc_t::index_t::value> index_t;
            typedef typename boost::mpl::has_key<ij_caches_map_t, index_t>::type type;
            static const bool value = type::value;
        };

        /**
         * @brief metafunction that determines if an accessor is accessed via kcache register set
         */
        template <typename Accessor>
        struct accessor_from_kcache_reg {
            typedef typename boost::remove_reference<Accessor>::type acc_t;

            GRIDTOOLS_STATIC_ASSERT((is_accessor<acc_t>::value), GT_INTERNAL_ERROR);
            typedef static_uint<acc_t::index_t::value> index_t;
            typedef typename boost::mpl::has_key<k_caches_map_t, index_t>::type type;
            static const bool value = type::value;
        };

        /** @brief return a value that was cached
         * specialization where cache goes via shared memory
         */
        template <typename ReturnType, typename Accessor>
        GT_FUNCTION typename boost::enable_if<accessor_from_shared_mem<Accessor>, ReturnType>::type
        get_cache_value_impl(Accessor const &accessor_) const {
            typedef typename boost::remove_const<typename boost::remove_reference<Accessor>::type>::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor<acc_t>::value), GT_INTERNAL_ERROR);

            //        assert(m_pshared_iterate_domain);
            // retrieve the ij cache from the fusion tuple and access the element required give the current thread
            // position within
            // the block and the offsets of the accessor
            return m_pshared_iterate_domain->template get_ij_cache<static_uint<acc_t::index_t::value>>().at<0>(
                m_thread_pos, accessor_);
        }

        /** @brief return a value that was cached
         * specialization where cache is explicitly disabled by user
         */
        template <typename ReturnType, typename Accessor>
        GT_FUNCTION
            typename boost::enable_if<boost::mpl::has_key<bypass_caches_set_t,
                                          static_uint<boost::remove_reference<Accessor>::type::index_type::value>>,
                ReturnType>::type
            get_cache_value_impl(Accessor const &accessor_) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            return super::template get_value<Accessor, void * RESTRICT>(
                accessor_, aux::get_data_pointer<Accessor>(super::local_domain, accessor_));
        }

        /** @brief return a value that was cached
         * specialization where cache goes via kcache register set
         *
         */
        template <typename ReturnType, typename Accessor>
        GT_FUNCTION typename boost::enable_if<accessor_from_kcache_reg<Accessor>, ReturnType>::type
        get_cache_value_impl(Accessor const &accessor_) {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            return m_iterate_domain_cache.template get_k_cache<static_uint<Accessor::index_t::value>>().at(accessor_);
        }

        /** @brief return a the value in memory pointed to by an accessor
         * specialization where the accessor points to an arg which is readonly for all the ESFs in all MSSs
         * Value is read via texture system
         */
        template <typename ReturnType, typename Accessor, typename StorageType>
        GT_FUNCTION typename boost::enable_if<typename accessor_read_from_texture<Accessor>::type, ReturnType>::type
        get_value_impl(StorageType *RESTRICT storage_pointer, int_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
#if __CUDA_ARCH__ >= 350
            // on Kepler use ldg to read directly via read only cache
            return __ldg(storage_pointer + pointer_offset);
#else
            return *(storage_pointer + pointer_offset);
#endif
        }

        /** @brief return a the value in memory pointed to by an accessor
         * specialization where the accessor points to an arg which is not readonly for all the ESFs in all MSSs
         */
        template <typename ReturnType, typename Accessor, typename StorageType>
        GT_FUNCTION typename boost::disable_if<typename accessor_read_from_texture<Accessor>::type, ReturnType>::type
        get_value_impl(StorageType *RESTRICT storage_pointer, int_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            return *(storage_pointer + pointer_offset);
        }

        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template slide_caches<IterationPolicy>();
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy, typename Grid>
        GT_FUNCTION void fill_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "error");

            m_iterate_domain_cache.template fill_caches<IterationPolicy>(*this, klevel, grid);
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy, typename Grid>
        GT_FUNCTION void flush_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "error");

            m_iterate_domain_cache.template flush_caches<IterationPolicy>(*this, klevel, grid);
        }

        /**
         * Final flush of the of the kcaches. After the iteration over k is done, we still need to flush the remaining
         * k levels of the cache with k > 0 (<0) for the backward (forward) iteration policy
         * \tparam IterationPolicy forward: backward
         */
        template <typename IterationPolicy>
        GT_FUNCTION void final_flush() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template final_flush<IterationPolicy>(*this);
        }

        /**
         * Initial fill of the of the kcaches. Before the iteration over k starts, we need to prefill the k level
         * of the cache with k > 0 (<0) for the forward (backward) iteration policy
         * \tparam IterationPolicy forward: backward
         */
        template <typename IterationPolicy>
        GT_FUNCTION void begin_fill() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            m_iterate_domain_cache.template begin_fill<IterationPolicy>(*this);
        }

      private:
        // array storing the (i,j) position of the current thread within the block
        array<int, 2> m_thread_pos;
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
