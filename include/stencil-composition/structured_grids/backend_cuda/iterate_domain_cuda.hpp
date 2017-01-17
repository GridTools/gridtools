/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <boost/type_traits/is_arithmetic.hpp>
#include "../../iterate_domain.hpp"
#include "../../iterate_domain_metafunctions.hpp"
#include "../../backend_cuda/iterate_domain_cache.hpp"
#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../../common/cuda_type_traits.hpp"

namespace gridtools {

    template < typename T >
    struct accessor_return_type;

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    class iterate_domain_cuda
        : public IterateDomainBase< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > > // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal error: wrong type");

        typedef IterateDomainBase< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > > super;
        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename local_domain_t::esf_args local_domain_args_t;

      public:
        template < typename Accessor >
        struct accessor_return_type {
            typedef typename super::template accessor_return_type< Accessor >::type type;
        };

        /**
         * metafunction that computes the return type of all operator() of an accessor.
         *
         * If the temaplate argument is not an accessor ::type is mpl::void_
         *
         */

        typedef typename super::data_pointer_array_t data_pointer_array_t;
        typedef typename super::strides_cached_t strides_cached_t;

        typedef typename super::iterate_domain_cache_t iterate_domain_cache_t;
        typedef typename super::readonly_args_indices_t readonly_args_indices_t;

      private:
        // TODO there are two instantiations of these type.. Fix this
        typedef shared_iterate_domain< data_pointer_array_t,
            strides_cached_t,
            typename IterateDomainArguments::max_extent_t,
            typename iterate_domain_cache_t::ij_caches_tuple_t > shared_iterate_domain_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::k_caches_map_t k_caches_map_t;
        typedef typename iterate_domain_cache_t::bypass_caches_set_t bypass_caches_set_t;
        typedef typename super::reduction_type_t reduction_type_t;

        using super::get_value;
        using super::get_data_pointer;

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

        GT_FUNCTION
        uint_t thread_position_x() const { return threadIdx.x; }

        template < int_t minus, int_t plus >
        GT_FUNCTION uint_t thread_position_y() const {
            return threadIdx.y;
        }

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template < typename Extent >
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

        /**
         * @brief determines whether the current (i) position + an offset is within the block size
         */
        template < int_t minus, int_t plus >
        GT_FUNCTION bool is_thread_in_domain_x(const int_t i_offset) const {
            return m_thread_pos[0] + i_offset >= minus && m_thread_pos[0] + i_offset < (int)m_block_size_i + plus;
        }

        /**
         * @brief determines whether the current (j) position is within the block size
         */
        template < int_t minus, int_t plus >
        GT_FUNCTION bool is_thread_in_domain_y(const int_t j_offset) const {
            return m_thread_pos[1] + j_offset >= minus && m_thread_pos[1] + j_offset < (int)m_block_size_j + plus;
        }

        GT_FUNCTION
        uint_t block_size_i() { return m_block_size_i; }
        GT_FUNCTION
        uint_t block_size_j() { return m_block_size_j; }

        GT_FUNCTION
        void set_shared_iterate_domain_pointer_impl(shared_iterate_domain_t *ptr) { m_pshared_iterate_domain = ptr; }

        GT_FUNCTION
        data_pointer_array_t const &RESTRICT data_pointer_impl() const {
            //        assert(m_pshared_iterate_domain);
            return m_pshared_iterate_domain->data_pointer();
        }

        GT_FUNCTION
        data_pointer_array_t &RESTRICT data_pointer_impl() {
            //        assert(m_pshared_iterate_domain);
            return m_pshared_iterate_domain->data_pointer();
        }

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

        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment_impl() {
            if (Coordinate != 0 && Coordinate != 1)
                return;
            m_thread_pos[Coordinate] += Execution::value;
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void increment_impl(const int_t steps) {
            if (Coordinate != 0 && Coordinate != 1)
                return;
            m_thread_pos[Coordinate] += steps;
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize_impl() {
            if (Coordinate == 0)
                m_thread_pos[Coordinate] = threadIdx.x;
            else if (Coordinate == 1)
                m_thread_pos[Coordinate] = threadIdx.y;
        }

        /** @brief metafunction that determines if an arg is pointing to a field which is read only by all ESFs
        */
        template < typename Accessor >
        struct accessor_points_to_readonly_arg {

            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");

            typedef typename boost::mpl::at< local_domain_args_t,
                boost::mpl::integral_c< int, Accessor::index_type::value > >::type arg_t;

            typedef typename boost::mpl::has_key< readonly_args_indices_t,
                boost::mpl::integral_c< int, arg_index< arg_t >::value > >::type type;
        };

        /**
        * @brief metafunction that determines if an accessor has to be read from texture memory
        */
        template < typename Accessor >
        struct accessor_read_from_texture {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_< typename accessor_points_to_readonly_arg< Accessor >::type,
                    typename boost::mpl::not_< typename boost::mpl::has_key< bypass_caches_set_t,
                        static_uint< Accessor::index_type::value > >::type                        // mpl::has_key
                                               >::type                                            // mpl::not,
                    >::type,                                                                      // mpl::(inner)and_
                typename is_texture_type< typename accessor_return_type< Accessor >::type >::type // is_texture_type
                >::type type;
        };

        /**
        * @brief metafunction that determines if an accessor is accessed via shared memory
        */
        template < typename Accessor >
        struct accessor_from_shared_mem {
            typedef typename boost::remove_reference< Accessor >::type acc_t;

            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Wrong type");
            typedef static_uint< acc_t::index_type::value > index_t;
            typedef typename boost::mpl::has_key< ij_caches_map_t, index_t >::type type;
            static const bool value = type::value;
        };

        /**
        * @brief metafunction that determines if an accessor is accessed via kcache register set
        */
        template < typename Accessor >
        struct accessor_from_kcache_reg {
            typedef typename boost::remove_reference< Accessor >::type acc_t;

            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Wrong type");
            typedef static_uint< acc_t::index_type::value > index_t;
            typedef typename boost::mpl::has_key< k_caches_map_t, index_t >::type type;
            static const bool value = type::value;
        };

        /** @brief return a value that was cached
        * specialization where cache goes via shared memory
        */
        template < typename ReturnType, typename Accessor >
        GT_FUNCTION typename boost::enable_if< accessor_from_shared_mem< Accessor >, ReturnType >::type
        get_cache_value_impl(Accessor const &accessor_) const {
            typedef typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Wrong type");

            //        assert(m_pshared_iterate_domain);
            // retrieve the ij cache from the fusion tuple and access the element required give the current thread
            // position within
            // the block and the offsets of the accessor
            return m_pshared_iterate_domain->template get_ij_cache< static_uint< acc_t::index_type::value > >().at(
                m_thread_pos, accessor_);
        }

        /** @brief return a value that was cached
        * specialization where cache is explicitly disabled by user
        */
        template < typename ReturnType, typename Accessor >
        GT_FUNCTION typename boost::enable_if<
            boost::mpl::has_key< bypass_caches_set_t,
                static_uint< boost::remove_reference< Accessor >::type::index_type::value > >,
            ReturnType >::type
        get_cache_value_impl(Accessor const &accessor_) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");
            return super::template get_value< Accessor, void * RESTRICT >(
                accessor_, super::template get_data_pointer< Accessor >(accessor_));
        }

        /** @brief return a value that was cached
        * specialization where cache goes via kcache register set
        *
        */
        template < typename ReturnType, typename Accessor >
        GT_FUNCTION typename boost::enable_if< accessor_from_kcache_reg< Accessor >, ReturnType >::type
        get_cache_value_impl(Accessor const &accessor_) {
            typedef typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Wrong type");

            // TODO KCACHE PROTECT no IJ accesses here

            // retrieve the k cache from the fusion tuple and access the element required give the current thread
            // position within
            // the block and the offsets of the accessor
            //            return
            return m_iterate_domain_cache.template get_k_cache< static_uint< acc_t::index_type::value > >().at(
                accessor_);
        }

        /** @brief return a the value in memory pointed to by an accessor
        * specialization where the accessor points to an arg which is readonly for all the ESFs in all MSSs
        * Value is read via texture system
        */
        template < typename ReturnType, typename Accessor, typename StoragePointer >
        GT_FUNCTION typename boost::enable_if< typename accessor_read_from_texture< Accessor >::type, ReturnType >::type
        get_value_impl(StoragePointer RESTRICT &storage_pointer, const uint_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");
#if __CUDA_ARCH__ >= 350
            // on Kepler use ldg to read directly via read only cache
            return __ldg(storage_pointer + pointer_offset);
#else
            return super::template get_gmem_value< ReturnType >(storage_pointer, pointer_offset);
#endif
        }

        /** @brief return a the value in memory pointed to by an accessor
        * specialization where the accessor points to an arg which is not readonly for all the ESFs in all MSSs
        */
        template < typename ReturnType, typename Accessor, typename StoragePointer >
        GT_FUNCTION
            typename boost::disable_if< typename accessor_read_from_texture< Accessor >::type, ReturnType >::type
            get_value_impl(StoragePointer RESTRICT &storage_pointer, const uint_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");
            return super::template get_gmem_value< ReturnType >(storage_pointer, pointer_offset);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            m_iterate_domain_cache.template slide_caches< IterationPolicy >();
        }

        template < typename IterationPolicy >
        GT_FUNCTION void fill_caches() {
            // TODO KCACHE do not execute if no flush caches and/or not in iteration policy
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            m_iterate_domain_cache.template fill_caches< IterationPolicy >(*this);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void flush_caches() {
            // TODO KCACHE do not execute if no flush caches and/or not in iteration policy
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            m_iterate_domain_cache.template flush_caches< IterationPolicy >(*this);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void final_flush() {
            // TODO KCACHE do not execute if no flush caches and/or not in iteration policy
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            m_iterate_domain_cache.template final_flush< IterationPolicy >(*this);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void begin_fill() {
            // TODO KCACHE do not execute if no flush caches and/or not in iteration policy
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            m_iterate_domain_cache.template begin_fill< IterationPolicy >(*this);
        }

      private:
        // array storing the (i,j) position of the current thread within the block
        array< int, 2 > m_thread_pos;
    };

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > >
        : public boost::mpl::true_ {};

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_positional_iterate_domain< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > >
        : is_positional_iterate_domain<
              IterateDomainBase< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > > > {};
}
