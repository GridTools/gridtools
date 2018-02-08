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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include "common/gt_assert.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/iterate_domain_metafunctions.hpp"
#include "stencil-composition/reductions/iterate_domain_reduction.hpp"

namespace gridtools {

    template < typename IterateDomainArguments >
    class iterate_domain_mic;

    template < typename IterateDomainArguments >
    struct iterate_domain_backend_id< iterate_domain_mic< IterateDomainArguments > > {
        using type = enumtype::enum_type< enumtype::platform, enumtype::Mic >;
    };

    namespace advanced {
        template < typename IDomain >
        inline typename IDomain::data_ptr_cached_t &RESTRICT get_iterate_domain_data_pointer(IDomain &id);
    } // namespace advanced

    /**
     * @brief iterate domain class for the MIC backend
     */
    template < typename IterateDomainArguments >
    class iterate_domain_mic : public iterate_domain_reduction< IterateDomainArguments > {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);

        using iterate_domain_arguments_t = IterateDomainArguments;
        using local_domain_t = typename iterate_domain_arguments_t::local_domain_t;
        using iterate_domain_reduction_t = iterate_domain_reduction< iterate_domain_arguments_t >;
        using reduction_type_t = typename iterate_domain_reduction_t::reduction_type_t;
        using grid_traits_t = typename iterate_domain_arguments_t::grid_traits_t;
        using backend_traits_t = backend_traits_from_id< enumtype::Mic >;
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), GT_INTERNAL_ERROR);

        //***************** end of internal type definitions
      public:
        //***************** types exposed in API
        using readonly_args_indices_t =
            typename compute_readonly_args_indices< typename iterate_domain_arguments_t::esf_sequence_t >::type;
        using esf_args_t = typename local_domain_t::esf_args;
        //*****************

        /**
         * metafunction that determines if a given accessor is associated with an placeholder holding a data field
         */
        template < typename Accessor >
        struct accessor_holds_data_field {
            using type = typename aux::accessor_holds_data_field< Accessor, iterate_domain_arguments_t >::type;
        };

        /**
         * metafunction that computes the return type of all operator() of an accessor.
         *
         * If the temaplate argument is not an accessor ::type is mpl::void_
         *
         */
        template < typename Accessor >
        struct accessor_return_type {
            using type = typename ::gridtools::accessor_return_type_impl< Accessor, iterate_domain_arguments_t >::type;
        };

        using storage_info_ptrs_t = typename local_domain_t::storage_info_ptr_fusion_list;
        using data_ptrs_map_t = typename local_domain_t::data_ptr_fusion_map;

        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< storage_info_ptrs_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< data_ptrs_map_t >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS =
            total_storages< typename local_domain_t::storage_wrapper_list_t, N_STORAGES >::type::value;

        using data_ptr_cached_t = data_ptr_cached< typename local_domain_t::storage_wrapper_list_t >;
        using strides_cached_t = strides_cached< N_META_STORAGES - 1, storage_info_ptrs_t >;
        // *************** end of type definitions **************

      protected:
        // *********************** members **********************
        local_domain_t const &local_domain;
        data_ptr_cached_t m_data_pointer;
        strides_cached_t m_strides;
        int_t m_i_block_index, m_j_block_index, m_k_block_index, m_i_block_base, m_j_block_base;
        int_t m_prefetch_distance;
        // ******************* end of members *******************

        // helper class for index array generation
        struct index_getter {
            index_getter(iterate_domain_mic const &it_domain, array< int_t, N_META_STORAGES > &index_array)
                : m_it_domain(it_domain), m_index_array(index_array) {}

            template < class StorageInfoIndex >
            void operator()(StorageInfoIndex) {
                const auto *storage_info =
                    boost::fusion::at< StorageInfoIndex >(m_it_domain.local_domain.m_local_storage_info_ptrs);
                using storage_info_t = typename boost::remove_pointer< decltype(storage_info) >::type;
                const int_t stride_i = m_it_domain.stride< storage_info_t, 0 >();
                const int_t stride_j = m_it_domain.stride< storage_info_t, 1 >();
                const int_t stride_k = m_it_domain.stride< storage_info_t, 2 >();
                m_index_array[StorageInfoIndex::value] = m_it_domain.m_i_block_index * stride_i +
                                                         m_it_domain.m_j_block_index * stride_j +
                                                         m_it_domain.m_k_block_index * stride_k;
            }

          private:
            array< int_t, N_META_STORAGES > &m_index_array;
            iterate_domain_mic const &m_it_domain;
        };

      public:
        GT_FUNCTION
        iterate_domain_mic(local_domain_t const &local_domain, reduction_type_t const &reduction_initial_value)
            : iterate_domain_reduction_t(reduction_initial_value), local_domain(local_domain), m_i_block_index(0),
              m_j_block_index(0), m_k_block_index(0), m_prefetch_distance(0) {
            // assign storage pointers
            boost::fusion::for_each(local_domain.m_local_data_ptrs,
                assign_storage_ptrs< backend_traits_t,
                                        data_ptr_cached_t,
                                        local_domain_t,
                                        block_size< 0, 0, 0 >,
                                        grid_traits_t >(m_data_pointer, local_domain.m_local_storage_info_ptrs));
            // assign stride pointers
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                assign_strides< backend_traits_t, strides_cached_t, local_domain_t, block_size< 0, 0, 0 > >(m_strides));
        }

        GT_FUNCTION
        data_ptr_cached_t const &RESTRICT data_pointer() const { return m_data_pointer; }

        GT_FUNCTION void set_block_base(int_t i_block_base, int_t j_block_base) {
            m_i_block_base = i_block_base;
            m_j_block_base = j_block_base;
        }

        GT_FUNCTION void set_i_block_index(int_t i) { m_i_block_index = i; }
        GT_FUNCTION void set_j_block_index(int_t j) { m_j_block_index = j; }
        GT_FUNCTION void set_k_block_index(int_t k) { m_k_block_index = k; }

        GT_FUNCTION array< int_t, N_META_STORAGES > index() const {
            using index_range = boost::mpl::range_c< int_t, 0, N_META_STORAGES >;
            array< int_t, N_META_STORAGES > index;
            index_getter ig(*this, index);
            boost::mpl::for_each< index_range >(ig);
            return index;
        }

        GT_FUNCTION void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        template < typename T >
        GT_FUNCTION void info(T const &x) const {
            local_domain.info(x);
        }

        template < typename Accessor, typename StoragePointer, bool DirectGMemAccess = false >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

        template < typename Accessor >
        GT_FUNCTION
            typename boost::disable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            using index_t = typename Accessor::index_t;
            using storage_info_t = typename local_domain_t::template get_data_store< index_t >::type::storage_info_t;

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions <= storage_info_t::layout_t::masked_length,
                "requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            using acc_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Using EVAL is only allowed for an accessor type");
            return m_data_pointer.template get< index_t::value >()[0];
        }

        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            using index_t = typename Accessor::index_t;
            using arg_t = typename local_domain_t::template get_arg< index_t >::type;
            using storage_wrapper_t =
                typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type;
            using data_store_t = typename storage_wrapper_t::data_store_t;
            using storage_info_t = typename storage_wrapper_t::storage_info_t;
            using data_t = typename storage_wrapper_t::data_t;
            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions == storage_info_t::layout_t::masked_length + 2,
                "The dimension of the data_store_field accessor must be equals to storage dimension + 2 (component and "
                "snapshot)");

            const int_t idx = get_datafield_offset< data_store_t >::get(accessor);
            assert(
                idx < data_store_t::num_of_storages && "Out of bounds access when accessing data store field element.");

            return m_data_pointer.template get< index_t::value >()[idx];
        }

        template < typename Container, typename Tuple, uint_t... Ids >
        GT_FUNCTION auto static tuple_to_container(
            Container &&container_, Tuple const &tuple_, gt_integer_sequence< uint_t, Ids... >)
            -> decltype(container_(boost::fusion::at_c< Ids >(tuple_)...)) {
            return container_(boost::fusion::at_c< Ids >(tuple_)...);
        }

        template < typename Acc, typename... Args >
        using ret_t = typename boost::remove_reference< decltype(tuple_to_container(
            std::declval< typename get_storage_accessor< local_domain_t, Acc >::type::storage_t::data_t >(),
            std::declval< global_accessor_with_arguments< Acc, Args... > >().get_arguments(),
            make_gt_integer_sequence< uint_t, sizeof...(Args) >())) >::type;

        template < ushort_t Coordinate, typename Accessor >
        GT_FUNCTION uint_t get_storage_dim(Accessor) const {
            GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, GT_INTERNAL_ERROR);
            typedef typename Accessor::index_type index_t;
            typedef typename local_domain_t::template get_data_store< index_t >::type::storage_info_t storage_info_t;
            typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const storage_info_t * >::type::pos storage_info_index_t;
            return boost::fusion::at< storage_info_index_t >(local_domain.m_local_storage_info_ptrs)
                ->template dim< Coordinate >();
        }

        template < uint_t I >
        GT_FUNCTION typename accessor_return_type< global_accessor< I > >::type operator()(
            global_accessor< I > const &accessor) {
            using return_t = typename accessor_return_type< global_accessor< I > >::type;
            using index_t = typename global_accessor< I >::index_t;
            return *static_cast< return_t * >(m_data_pointer.template get< index_t::value >()[0]);
        }

        template < typename Accessor >
        GT_FUNCTION typename boost::disable_if<
            boost::mpl::or_< boost::mpl::not_< is_accessor< Accessor > >, is_global_accessor< Accessor > >,
            typename accessor_return_type< Accessor >::type >::type
        operator()(Accessor const &accessor) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dimensions > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");

            return get_value(accessor, get_data_pointer(accessor));
        }

        template < typename... Arguments, template < typename... Args > class Expression >
        GT_FUNCTION auto operator()(Expression< Arguments... > const &arg)
            -> decltype(expressions::evaluation::value(*this, arg)) {
            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Arguments... > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }

        template < typename Argument, template < typename Arg1, int Arg2 > class Expression, int exponent >
        GT_FUNCTION auto operator()(Expression< Argument, exponent > const &arg)
            -> decltype(expressions::evaluation::value((*this), arg)) {

            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Argument, exponent > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }

        template < typename Acc, typename... Args >
        GT_FUNCTION auto operator()(global_accessor_with_arguments< Acc, Args... > const &accessor) const
            -> ret_t< Acc, Args... > {
            typedef typename Acc::index_t index_t;
            auto storage_ = boost::fusion::at< index_t >(local_domain.m_local_data_ptrs).second;

            return tuple_to_container(
                **storage_.data(), accessor.get_arguments(), make_gt_integer_sequence< uint_t, sizeof...(Args) >());
        }

        GT_FUNCTION
        int_t i() const { return m_i_block_base + m_i_block_index; }

        GT_FUNCTION
        int_t j() const { return m_j_block_base + m_j_block_index; }

        GT_FUNCTION
        int_t k() const { return m_k_block_index; }

      private:
        GT_FUNCTION
        data_ptr_cached_t &RESTRICT data_pointer() { return m_data_pointer; }

        friend data_ptr_cached_t &RESTRICT advanced::get_iterate_domain_data_pointer< iterate_domain_mic >(
            iterate_domain_mic &);

        template < typename StorageInfo, int_t Coordinate >
        GT_FUNCTION int_t stride() const {
            using layout_t = typename StorageInfo::layout_t;
            using layout_vector_t = typename layout_t::static_layout_vector;
            constexpr int_t layout_max =
                boost::mpl::deref< typename boost::mpl::max_element< layout_vector_t >::type >::type::value;
            constexpr int_t layout_val =
                Coordinate < layout_t::masked_length
                    ? layout_t::template at< (Coordinate < layout_t::masked_length ? Coordinate : 0) >()
                    : -1;
            constexpr bool is_max = layout_val == layout_max;
            constexpr bool is_masked = layout_val == -1;

            constexpr int_t storage_info_index = boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const StorageInfo * >::type::pos::value;
            return is_masked ? 0 : is_max ? 1 : m_strides.template get< storage_info_index >()[layout_val];
        }

        template < typename StorageInfo, int_t Coordinate >
        GT_FUNCTION int_t base_offset() const {
            // block offset in i- and j-dimension
            constexpr bool is_tmp =
                boost::mpl::at< typename local_domain_t::storage_info_tmp_info_t, StorageInfo >::type::value;
            constexpr int_t halo_i = StorageInfo::halo_t::template at< 0 >();
            constexpr int_t halo_j = StorageInfo::halo_t::template at< 1 >();
            const int_t block_offset = Coordinate == 0 ? (is_tmp ? halo_i : m_i_block_base)
                                                       : Coordinate == 1 ? (is_tmp ? halo_j : m_j_block_base) : 0;

            // index offset in i-, j- and k-dimension
            const int_t index_offset = Coordinate == 0 ? m_i_block_index : Coordinate == 1
                                                                               ? m_j_block_index
                                                                               : Coordinate == 2 ? m_k_block_index : 0;
            return block_offset + index_offset;
        }

        template < typename StorageInfo,
            typename Accessor,
            int_t Coordinate = StorageInfo::layout_t::masked_length - 1 >
        GT_FUNCTION typename std::enable_if< (Coordinate >= 0), int_t >::type compute_offset(
            Accessor const &accessor) const {
            // base index offset
            const int_t index_offset = base_offset< StorageInfo, Coordinate >();

            // accessor offset in all dimensions
            constexpr int_t accessor_index =
                is_array< Accessor >::value ? Coordinate : Accessor::n_dimensions - 1 - Coordinate;
            const int_t accessor_offset = accessor.template get< accessor_index >();

            // total offset
            const int_t offset = (index_offset + accessor_offset) * stride< StorageInfo, Coordinate >();

            // recursively add offsets of lower dimensions
            return offset + compute_offset< StorageInfo, Accessor, Coordinate - 1 >(accessor);
        }

        template < typename StorageInfo, typename Accessor, int_t Coordinate >
        GT_FUNCTION typename std::enable_if< (Coordinate == -1), int_t >::type compute_offset(Accessor const &) const {
            // base case of recursive offset computation
            return 0;
        }
    };

    template < typename IterateDomainArguments >
    template < typename Accessor, typename StoragePointer, bool DirectGMemAccess >
    GT_FUNCTION typename iterate_domain_mic< IterateDomainArguments >::template accessor_return_type< Accessor >::type
    iterate_domain_mic< IterateDomainArguments >::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {

        using return_t =
            typename iterate_domain_mic< IterateDomainArguments >::template accessor_return_type< Accessor >::type;

        // getting information about the storage
        using index_t = typename Accessor::index_t;
        using arg_t = typename local_domain_t::template get_arg< index_t >::type;

        using storage_wrapper_t =
            typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type;
        using storage_info_t = typename storage_wrapper_t::storage_info_t;
        using data_t = typename storage_wrapper_t::data_t;

        // this index here describes the position of the storage info in the m_index array (can be different to the
        // storage info id)
        using storage_info_index_t = typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
            const storage_info_t * >::type::pos;

        const storage_info_t *storage_info =
            boost::fusion::at< storage_info_index_t >(local_domain.m_local_storage_info_ptrs);

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

        assert(storage_pointer);
        data_t *RESTRICT real_storage_pointer = static_cast< data_t * >(storage_pointer);
        assert(real_storage_pointer);

        const int_t pointer_offset = compute_offset< storage_info_t >(accessor);

#ifndef NDEBUG
        GTASSERT((pointer_oob_check< backend_traits_t, block_size< 0, 0, 0 >, local_domain_t, arg_t, grid_traits_t >(
            storage_info, real_storage_pointer, pointer_offset)));
#endif

#ifdef __SSE__
        if (m_prefetch_distance != 0) {
            const int_t prefetch_offset = m_prefetch_distance * stride< storage_info_t, 2 >();
            _mm_prefetch(
                reinterpret_cast< const char * >(&real_storage_pointer[pointer_offset + prefetch_offset]), _MM_HINT_T1);
        }
#endif
        return real_storage_pointer[pointer_offset];
    }

    template < typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain_mic< IterateDomainArguments > > : boost::mpl::true_ {};

    template < typename IterateDomainArguments >
    struct is_positional_iterate_domain< iterate_domain_mic< IterateDomainArguments > > : boost::mpl::true_ {};

} // namespace gridtools
