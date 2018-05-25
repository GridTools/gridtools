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

#include <functional>

#include <boost/fusion/functional/invocation/invoke.hpp>

#include "common/gt_assert.hpp"
#include "common/generic_metafunctions/for_each.hpp"
#include "common/generic_metafunctions/meta.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "stencil-composition/iterate_domain_aux.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/iterate_domain_metafunctions.hpp"
#include "stencil-composition/reductions/iterate_domain_reduction.hpp"
#include "stencil-composition/offset_computation.hpp"

namespace gridtools {

    template < typename IterateDomainArguments >
    class iterate_domain_mic;

    template < typename IterateDomainArguments >
    struct iterate_domain_backend_id< iterate_domain_mic< IterateDomainArguments > > {
        using type = enumtype::enum_type< enumtype::platform, enumtype::Mic >;
    };

    /**
     * @brief Iterate domain class for the MIC backend.
     */
    template < typename IterateDomainArguments >
    class iterate_domain_mic : public iterate_domain_reduction< IterateDomainArguments > {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), GT_INTERNAL_ERROR);

        using iterate_domain_reduction_t = iterate_domain_reduction< IterateDomainArguments >;
        using reduction_type_t = typename iterate_domain_reduction_t::reduction_type_t;
        using grid_traits_t = typename IterateDomainArguments::grid_traits_t;
        using backend_traits_t = backend_traits_from_id< enumtype::Mic >;

        using esf_sequence_t = typename IterateDomainArguments::esf_sequence_t;
        using cache_sequence_t = typename IterateDomainArguments::cache_sequence_t;

        /* meta function to get storage info index in local domain */
        template < typename StorageInfo >
        using local_domain_storage_index =
            typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list, const StorageInfo * >::type::pos;

        /* meta function to check if a storage info belongs to a temporary field */
        template < typename StorageInfo >
        using storage_is_tmp =
            typename boost::mpl::at< typename local_domain_t::storage_info_tmp_info_t, StorageInfo >::type;

        /* meta function to get the storage info type corresponding to an accessor */
        template < typename Accessor >
        using storage_info_from_accessor =
            typename local_domain_t::template get_data_store< typename Accessor::index_t >::type::storage_info_t;

        /* ij-cache types and meta functions */
        using ij_caches_t = typename boost::mpl::copy_if< cache_sequence_t, cache_is_type< IJ > >::type;
        using ij_cache_indices_t =
            typename boost::mpl::transform< ij_caches_t, cache_to_index< boost::mpl::_1, local_domain_t > >::type;
        using ij_cache_indexset_t = typename boost::mpl::fold< ij_cache_indices_t,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type;
        template < typename Accessor >
        using accessor_is_ij_cached = typename accessor_is_cached< Accessor, ij_cache_indexset_t >::type;

      public:
        //***************** types exposed in API
        using readonly_args_indices_t =
            typename compute_readonly_args_indices< typename IterateDomainArguments::esf_sequence_t >::type;
        using esf_args_t = typename local_domain_t::esf_args;
        //*****************

        /**
         * @brief metafunction that determines if a given accessor is associated with an placeholder holding a data
         * field.
         */
        template < typename Accessor >
        struct accessor_holds_data_field {
            using type = typename aux::accessor_holds_data_field< Accessor, IterateDomainArguments >::type;
        };

        /**
         * @brief metafunction that computes the return type of all operator() of an accessor.
         * If the temaplate argument is not an accessor `type` is mpl::void_.
         */
        template < typename Accessor >
        struct accessor_return_type {
            using type = typename ::gridtools::accessor_return_type_impl< Accessor, IterateDomainArguments >::type;
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
        using array_index_t = array< int_t, N_META_STORAGES >;
        // *************** end of type definitions **************

      protected:
        // *********************** members **********************
        local_domain_t const &local_domain;
        data_ptr_cached_t m_data_pointer;
        strides_cached_t m_strides;
        int_t m_i_block_index;     /** Local i-index inside block. */
        int_t m_j_block_index;     /** Local j-index inside block. */
        int_t m_k_block_index;     /** Local/global k-index (no blocking along k-axis). */
        int_t m_i_block_base;      /** Global block start index along i-axis. */
        int_t m_j_block_base;      /** Global block start index along j-axis. */
        int_t m_prefetch_distance; /** Prefetching distance along k-axis, zero means no software prefetching. */
        bool m_enable_ij_caches;   /** Enables ij-caching. */
        // ******************* end of members *******************

        // helper class for index array generation, only needed for the index() function
        struct index_getter {
            index_getter(iterate_domain_mic const &it_domain, array_index_t &index_array)
                : m_it_domain(it_domain), m_index_array(index_array) {}

            template < class StorageInfoIndex >
            void operator()(StorageInfoIndex const &) const {
                using storage_info_ptrref_t =
                    typename boost::fusion::result_of::at< typename local_domain_t::storage_info_ptr_fusion_list,
                        StorageInfoIndex >::type;
                using storage_info_t = typename std::remove_const< typename std::remove_pointer<
                    typename std::remove_reference< storage_info_ptrref_t >::type >::type >::type;
                m_index_array[StorageInfoIndex::value] =
                    m_it_domain.compute_offset< storage_info_t >(accessor_base< storage_info_t::ndims >());
            }

          private:
            array< int_t, N_META_STORAGES > &m_index_array;
            iterate_domain_mic const &m_it_domain;
        };

      public:
        GT_FUNCTION
        iterate_domain_mic(local_domain_t const &local_domain, reduction_type_t const &reduction_initial_value)
            : iterate_domain_reduction_t(reduction_initial_value), local_domain(local_domain), m_i_block_index(0),
              m_j_block_index(0), m_k_block_index(0), m_i_block_base(0), m_j_block_base(0), m_prefetch_distance(0),
              m_enable_ij_caches(false) {
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

        /** @brief Returns the array of pointers to the raw data as const reference. */
        GT_FUNCTION
        data_ptr_cached_t const &RESTRICT data_pointer() const { return m_data_pointer; }

        /** @brief Sets the block start indices. */
        GT_FUNCTION void set_block_base(int_t i_block_base, int_t j_block_base) {
            m_i_block_base = i_block_base;
            m_j_block_base = j_block_base;
        }

        /** @brief Sets the local block index along the i-axis. */
        GT_FUNCTION void set_i_block_index(int_t i) { m_i_block_index = i; }
        /** @brief Sets the local block index along the j-axis. */
        GT_FUNCTION void set_j_block_index(int_t j) { m_j_block_index = j; }
        /** @brief Sets the local block index along the k-axis. */
        GT_FUNCTION void set_k_block_index(int_t k) { m_k_block_index = k; }

        /** @brief Returns the current data index at offset (0, 0, 0) per meta storage. */
        GT_FUNCTION array_index_t index() const {
            array_index_t index_array;
            for_each< GT_META_CALL(meta::make_indices_c, N_META_STORAGES) >(index_getter(*this, index_array));
            return index_array;
        }

        /** @brief Sets the software prefetching distance along k-axis. Zero means no software prefetching. */
        GT_FUNCTION void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        /** @brief Enables ij-caches. */
        GT_FUNCTION void enable_ij_caches() { m_enable_ij_caches = true; }

        template < typename T >
        GT_FUNCTION void info(T const &x) const {
            local_domain.info(x);
        }

        /**
         * @brief Returns the value of the memory at the given address, plus the offset specified by the arg
         * placeholder.
         *
         * @param accessor Accessor passed to the evaluator.
         * @param storage_pointer Pointer to the first element of the specific data field used.
        */
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

        /**
         * @brief Method returning the data pointer of an accessor.
         * Specialization for the accessor placeholders for standard storages.
         *
         * This method is enabled only if the current placeholder dimension does not exceed the number of space
         * dimensions of the storage class. I.e., if we are dealing with storages, not with storage lists or data fields
         * (see concepts page for definitions)
        */
        template < typename Accessor >
        GT_FUNCTION
            typename boost::disable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            using index_t = typename Accessor::index_t;
            using storage_info_t = storage_info_from_accessor< Accessor >;

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions <= storage_info_t::layout_t::masked_length,
                "requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            using acc_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Using EVAL is only allowed for an accessor type");
            return m_data_pointer.template get< index_t::value >()[0];
        }

        /**
         * @brief Method returning the data pointer of an accessor.
         * Specialization for the accessor placeholder for extended storages,
         * containg multiple snapshots of data fields with the same dimension and memory layout.
         *
         * This method is enabled only if the current placeholder dimension exceeds the number of space dimensions of
         * the storage class. I.e., if we are dealing with storage lists or data fields (see concepts page for
         * definitions).
        */
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

        /**
         * @brief Returns the dimension of the storage corresponding to the given accessor.
         * Useful to determine the loop bounds, when looping over a dimension from whithin a kernel.
         */
        template < ushort_t Coordinate, typename Accessor >
        GT_FUNCTION uint_t get_storage_dim(Accessor) const {
            GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, GT_INTERNAL_ERROR);
            using storage_info_t = storage_info_from_accessor< Accessor >;
            using storage_index_t = local_domain_storage_index< storage_info_t >;
            return boost::fusion::at< storage_index_t >(local_domain.m_local_storage_info_ptrs)
                ->template dim< Coordinate >();
        }

        /**
         * @brief Method called in the Do methods of the functors.
         * Specialization for the generic accessors placeholders.
        */
        template < uint_t I >
        GT_FUNCTION typename accessor_return_type< global_accessor< I > >::type operator()(
            global_accessor< I > const &accessor) {
            using return_t = typename accessor_return_type< global_accessor< I > >::type;
            using index_t = typename global_accessor< I >::index_t;
            return *static_cast< return_t * >(m_data_pointer.template get< index_t::value >()[0]);
        }

        /**
         * @brief Method called in the Do methods of the functors.
         * Specialization for the generic accessors placeholders with arguments.
        */
        template < typename Acc, typename... Args >
        GT_FUNCTION auto operator()(global_accessor_with_arguments< Acc, Args... > const &accessor) const /** @cond */
            GT_AUTO_RETURN(boost::fusion::invoke(
                std::cref(**boost::fusion::at< typename Acc::index_t >(local_domain.m_local_data_ptrs).second.data()),
                accessor.get_arguments())) /** @endcond */;

        /**
         * @brief Returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression).
         */
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

        /** @brief Global i-index. */
        GT_FUNCTION
        int_t i() const { return m_i_block_base + m_i_block_index; }

        /** @brief Global j-index. */
        GT_FUNCTION
        int_t j() const { return m_j_block_base + m_j_block_index; }

        /** @brief Global k-index. */
        GT_FUNCTION
        int_t k() const { return m_k_block_index; }

      private:
        /**
         * @brief Returns stride for a storage along the given axis.
         *
         * @tparam StorageInfo Storage info for which the strides should be returned.
         * @tparam Coordinate Axis/coordinate along which the stride is needed.
         */
        template < typename StorageInfo, int_t Coordinate >
        GT_FUNCTION int_t storage_stride() const {
            using storage_index_t = local_domain_storage_index< StorageInfo >;
            auto const &strides = m_strides.template get< storage_index_t::value >();
            return stride< StorageInfo, Coordinate >(strides);
        }

        /**
         * @brief Computes the global offset of a data access along the given axis.
         *
         * Computation includes the current index and the offsets stored in the accessor.
         * Version for i-axis.
         *
         * @tparam StorageInfo Storage info for which the offset should be computed.
         * @tparam Coordinate Axis/coordinate along which the offset should be computed.
         * @tparam Accessor An accessor type.
         *
         * @param accessor Accessor for which the offset should be returned.
         *
         * @return Global offset induced by current index and possibly the accessor along given axis.
         */
        template < typename StorageInfo, int_t Coordinate, typename Accessor >
        GT_FUNCTION typename std::enable_if< Coordinate == 0, int_t >::type coordinate_offset(
            Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp< StorageInfo >::value;
            constexpr int_t halo = StorageInfo::halo_t::template at< Coordinate >();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_i_block_base;
            return block_base + m_i_block_index + accessor_offset< Coordinate >(accessor);
        }

        /**
         * @brief Computes the global offset of a data access along the given axis.
         *
         * Computation includes the current index and the offsets stored in the accessor.
         * Version for j-axis.
         *
         * @tparam StorageInfo Storage info for which the offset should be computed.
         * @tparam Coordinate Axis/coordinate along which the offset should be computed.
         * @tparam Accessor An accessor type.
         *
         * @param accessor Accessor for which the offset should be returned.
         *
         * @return Global offset induced by current index and possibly the accessor along given axis.
         */
        template < typename StorageInfo, int_t Coordinate, typename Accessor >
        GT_FUNCTION typename std::enable_if< Coordinate == 1, int_t >::type coordinate_offset(
            Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp< StorageInfo >::value;
            constexpr int_t halo = StorageInfo::halo_t::template at< Coordinate >();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_j_block_base;
            return block_base + m_j_block_index + accessor_offset< Coordinate >(accessor);
        }

        /**
         * @brief Computes the global offset of a data access along the given axis.
         *
         * Computation includes the current index and the offsets stored in the accessor.
         * Version for k-axis.
         *
         * @tparam StorageInfo Storage info for which the offset should be computed.
         * @tparam Coordinate Axis/coordinate along which the offset should be computed.
         * @tparam Accessor An accessor type.
         *
         * @param accessor Accessor for which the offset should be returned.
         *
         * @return Global offset induced by current index and possibly the accessor along given axis.
         */
        template < typename StorageInfo, int_t Coordinate, typename Accessor >
        GT_FUNCTION typename std::enable_if< Coordinate == 2, int_t >::type coordinate_offset(
            Accessor const &accessor) const {
            // for ij-caches we simply ignore the block index and always access storage at k = 0
            const int_t block_index =
                (accessor_is_ij_cached< Accessor >::value && m_enable_ij_caches) ? 0 : m_k_block_index;
            return block_index + accessor_offset< Coordinate >(accessor);
        }

        /**
         * @brief Computes the global offset of a data access along the given axis.
         *
         * Computation includes the current index and the offsets stored in the accessor.
         * Version for higher dimensions.
         *
         * @tparam StorageInfo Storage info for which the offset should be computed.
         * @tparam Coordinate Axis/coordinate along which the offset should be computed.
         * @tparam Accessor An accessor type.
         *
         * @param accessor Accessor for which the offset should be returned.
         *
         * @return Global offset induced by current index and possibly the accessor along given axis.
         */
        template < typename StorageInfo, int_t Coordinate, typename Accessor >
        GT_FUNCTION constexpr typename std::enable_if< (Coordinate > 2), int_t >::type coordinate_offset(
            Accessor const &accessor) const {
            return accessor_offset< Coordinate >(accessor);
        }

        template < typename StorageInfo, typename Accessor, std::size_t... Coordinates >
        GT_FUNCTION int_t compute_offset_impl(Accessor const &accessor, gt_index_sequence< Coordinates... >) const {
            return accumulate(plus_functor(),
                (storage_stride< StorageInfo, Coordinates >() *
                                  coordinate_offset< StorageInfo, Coordinates >(accessor))...);
        }

        /**
         * @brief Computes the total linear data pointer offset for a storage when accessed with an accessor.
         *
         * @tparam StorageInfo Storage info of the storage to be accessed.
         * @tparam Accessor Accessor type.
         *
         * @param accessor Accessor for which the offset should be computed.
         *
         * @return A linear data pointer offset to access the data of a compatible storage.
         */
        template < typename StorageInfo, typename Accessor >
        GT_FUNCTION int_t compute_offset(Accessor const &accessor) const {
            using sequence_t = make_gt_index_sequence< StorageInfo::layout_t::masked_length >;
            return compute_offset_impl< StorageInfo >(accessor, sequence_t());
        }
    };

    /**
     * @brief Returns the value of the memory at the given address, plus the offset specified by the arg placeholder.
     * @param accessor Accessor passed to the evaluator.
     * @param storage_pointer Pointer to the first element of the specific data field used.
    */
    template < typename IterateDomainArguments >
    template < typename Accessor, typename StoragePointer >
    GT_FUNCTION typename iterate_domain_mic< IterateDomainArguments >::template accessor_return_type< Accessor >::type
    iterate_domain_mic< IterateDomainArguments >::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {
        // getting information about the storage
        using arg_t = typename local_domain_t::template get_arg< typename Accessor::index_t >::type;

        using storage_wrapper_t =
            typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type;
        using storage_info_t = typename storage_wrapper_t::storage_info_t;
        using data_t = typename storage_wrapper_t::data_t;

        using storage_index_t = local_domain_storage_index< storage_info_t >;

        const storage_info_t *storage_info =
            boost::fusion::at< storage_index_t >(local_domain.m_local_storage_info_ptrs);

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

        assert(storage_pointer);
        data_t *RESTRICT real_storage_pointer = static_cast< data_t * >(storage_pointer);
        assert(real_storage_pointer);

        const int_t pointer_offset = compute_offset< storage_info_t >(accessor);

        assert((pointer_oob_check< backend_traits_t, block_size< 0, 0, 0 >, local_domain_t, arg_t, grid_traits_t >(
            storage_info, real_storage_pointer, pointer_offset)));

#ifdef __SSE__
        if (m_prefetch_distance != 0) {
            const int_t prefetch_offset = m_prefetch_distance * storage_stride< storage_info_t, 2 >();
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
