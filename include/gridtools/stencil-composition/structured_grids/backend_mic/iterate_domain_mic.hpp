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

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/generic_metafunctions/meta.hpp"
#include "../../../common/gt_assert.hpp"
#include "../../../storage/data_field_view.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../iterate_domain_metafunctions.hpp"
#include "../../offset_computation.hpp"
#include "../../reductions/iterate_domain_reduction.hpp"

namespace gridtools {

    namespace _impl {
        /**
         * @brief compute thread offsets for temporaries
         *
         * Actually offsets are stored for each data_ptr, however only for temporaries they are non-zero.
         * We keep the zeros as it simplifies design, and will be cleaned up when we re-implement the temporaries.
         */
        template <typename LocalDomain, typename DataPtrsOffset>
        struct assign_data_ptr_offsets {
            LocalDomain const &m_local_domain;
            DataPtrsOffset &m_data_ptr_offsets;

            template <class Index, class Arg = GT_META_CALL(meta::at, (typename LocalDomain::esf_args, Index))>
            GT_FUNCTION enable_if_t<Arg::is_temporary> operator()(Index) const {
                using storage_info_t = typename Arg::data_store_t::storage_info_t;
                using storage_info_index_t =
                    GT_META_CALL(meta::st_position, (typename LocalDomain::storage_info_list, storage_info_t));
                const auto padded_total_length =
                    get<storage_info_index_t::value>(m_local_domain.m_local_padded_total_lengths);

                const int_t thread = omp_get_thread_num();
                const int_t total_threads = omp_get_max_threads();
                m_data_ptr_offsets[Index::value] = padded_total_length * thread / total_threads;
            }

            template <class Index, class Arg = GT_META_CALL(meta::at, (typename LocalDomain::esf_args, Index))>
            GT_FUNCTION enable_if_t<!Arg::is_temporary> operator()(Index) const {
                m_data_ptr_offsets[Index::value] = 0;
            }
        };

    } // namespace _impl

    /**
     * @brief Iterate domain class for the MIC backend.
     */
    template <typename IterateDomainArguments>
    class iterate_domain_mic : public iterate_domain_reduction<IterateDomainArguments> {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GRIDTOOLS_STATIC_ASSERT((is_local_domain<local_domain_t>::value), GT_INTERNAL_ERROR);

        using iterate_domain_reduction_t = iterate_domain_reduction<IterateDomainArguments>;
        using reduction_type_t = typename iterate_domain_reduction_t::reduction_type_t;
        using backend_traits_t = backend_traits_from_id<platform::mc>;

        using esf_sequence_t = typename IterateDomainArguments::esf_sequence_t;
        using cache_sequence_t = typename IterateDomainArguments::cache_sequence_t;

        /* meta function to check if a storage info belongs to a temporary field */
        template <typename StorageInfo>
        using storage_is_tmp = meta::st_contains<typename local_domain_t::tmp_storage_info_list, StorageInfo>;

        /* meta function to get the storage info type corresponding to an accessor */
        template <typename Accessor>
        using storage_info_from_accessor =
            typename local_domain_t::template get_arg<typename Accessor::index_t>::type::data_store_t::storage_info_t;

        /* ij-cache types and meta functions */
        using ij_caches_t = typename boost::mpl::copy_if<cache_sequence_t, cache_is_type<IJ>>::type;
        using ij_cache_indices_t =
            typename boost::mpl::transform<ij_caches_t, cache_to_index<boost::mpl::_1, local_domain_t>>::type;
        using ij_cache_indexset_t = typename boost::mpl::
            fold<ij_cache_indices_t, boost::mpl::set0<>, boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>>::type;
        template <typename Accessor>
        using accessor_is_ij_cached = typename accessor_is_cached<Accessor, ij_cache_indexset_t>::type;

      public:
        //***************** types exposed in API
        using readonly_args_indices_t =
            typename compute_readonly_args_indices<typename IterateDomainArguments::esf_sequence_t>::type;
        using esf_args_t = typename local_domain_t::esf_args;
        //*****************

        /**
         * @brief metafunction that determines if a given accessor is associated with an placeholder holding a data
         * field.
         */
        template <typename Accessor>
        struct accessor_holds_data_field {
            using type = typename aux::accessor_holds_data_field<Accessor, IterateDomainArguments>::type;
        };

        /**
         * @brief metafunction that computes the return type of all operator() of an accessor.
         * If the temaplate argument is not an accessor `type` is mpl::void_.
         */
        template <typename Accessor>
        struct accessor_return_type {
            using type = typename ::gridtools::accessor_return_type_impl<Accessor, IterateDomainArguments>::type;
        };

        using data_ptr_tuple_t = typename local_domain_t::data_ptr_tuple;

        // the number of different storage metadatas used in the current functor
        static constexpr auto N_META_STORAGES = meta::length<typename local_domain_t::storage_info_list>::value;
        // the number of storages  used in the current functor
        static constexpr auto N_STORAGES = meta::length<data_ptr_tuple_t>::value;

        using strides_t = typename local_domain_t::strides_tuple;
        using array_index_t = array<int_t, N_META_STORAGES>;
        // *************** end of type definitions **************

      protected:
        // *********************** members **********************
        local_domain_t const &local_domain;
        strides_t m_strides;
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

            template <class StorageInfoIndex>
            void operator()(StorageInfoIndex const &) const {
                using storage_info_t =
                    GT_META_CALL(meta::at, (typename local_domain_t::storage_info_list, StorageInfoIndex));
                m_index_array[StorageInfoIndex::value] =
                    m_it_domain.compute_offset<storage_info_t>(accessor_base<storage_info_t::ndims>());
            }

          private:
            iterate_domain_mic const &m_it_domain;
            array<int_t, N_META_STORAGES> &m_index_array;
        };

      private:
        using data_ptr_offsets_t = array<int, meta::length<decltype(local_domain.m_local_data_ptrs)>::value>;
        data_ptr_offsets_t m_data_ptr_offsets;

        /**
         * @brief get data pointer, taking into account a possible offset in case of temporaries
         */
        template <typename Accessor, typename Arg = typename get_arg_from_accessor<Accessor, local_domain_t>::type>
        GT_FUNCTION void *RESTRICT get_data_pointer(Accessor const &accessor) {
            static constexpr auto pos_in_args = meta::st_position<typename local_domain_t::esf_args, Arg>::value;
            return aux::get_data_pointer(local_domain, accessor) + m_data_ptr_offsets[pos_in_args];
        }

      public:
        GT_FUNCTION
        iterate_domain_mic(local_domain_t const &local_domain, reduction_type_t const &reduction_initial_value)
            : iterate_domain_reduction_t(reduction_initial_value), local_domain(local_domain),
              m_strides(local_domain.m_local_strides), m_i_block_index(0), m_j_block_index(0), m_k_block_index(0),
              m_i_block_base(0), m_j_block_base(0), m_prefetch_distance(0), m_enable_ij_caches(false) {
            gridtools::for_each<GT_META_CALL(meta::make_indices_for, esf_args_t)>(
                _impl::assign_data_ptr_offsets<local_domain_t, data_ptr_offsets_t>{local_domain, m_data_ptr_offsets});
        }

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
            for_each<GT_META_CALL(meta::make_indices_c, N_META_STORAGES)>(index_getter(*this, index_array));
            return index_array;
        }

        /** @brief Sets the software prefetching distance along k-axis. Zero means no software prefetching. */
        GT_FUNCTION void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        /** @brief Enables ij-caches. */
        GT_FUNCTION void enable_ij_caches() { m_enable_ij_caches = true; }

        /**
         * @brief Returns the value of the memory at the given address, plus the offset specified by the arg
         * placeholder.
         *
         * @param accessor Accessor passed to the evaluator.
         * @param storage_pointer Pointer to the first element of the specific data field used.
         */
        template <typename Accessor, typename StoragePointer>
        GT_FUNCTION typename accessor_return_type<Accessor>::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

        /**
         * @brief Method called in the Do methods of the functors.
         * Specialization for the global accessors placeholders.
         */
        template <uint_t I, class Res = typename accessor_return_type<global_accessor<I>>::type>
        GT_FUNCTION Res operator()(global_accessor<I> const &accessor) const {
            using index_t = typename global_accessor<I>::index_t;
            return *static_cast<Res *>(get<index_t::value>(local_domain.m_local_data_ptrs)[0]);
        }

        /**
         * @brief Method called in the Do methods of the functors.
         * Specialization for the global accessors placeholders with arguments.
         */
        template <typename Acc, typename... Args>
        GT_FUNCTION auto operator()(global_accessor_with_arguments<Acc, Args...> const &accessor) const /** @cond */
            GT_AUTO_RETURN(
                boost::fusion::invoke(std::cref(**get<Acc::index_t::value>(local_domain.m_local_data_ptrs).data()),
                    accessor.get_arguments())) /** @endcond */;

        /**
         * @brief Returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression).
         */
        template <typename Accessor>
        GT_FUNCTION typename boost::disable_if<
            boost::mpl::or_<boost::mpl::not_<is_accessor<Accessor>>, is_global_accessor<Accessor>>,
            typename accessor_return_type<Accessor>::type>::type
        operator()(Accessor const &accessor) {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");
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
        template <typename StorageInfo, int_t Coordinate>
        GT_FUNCTION int_t storage_stride() const {
            using storage_info_index_t =
                GT_META_CALL(meta::st_position, (typename local_domain_t::storage_info_list, StorageInfo));
            auto const &strides = get<storage_info_index_t::value>(m_strides);
            return stride<StorageInfo, Coordinate>(strides);
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
        template <typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION typename std::enable_if<Coordinate == 0, int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp<StorageInfo>::value;
            constexpr int_t halo = StorageInfo::halo_t::template at<Coordinate>();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_i_block_base;
            return block_base + m_i_block_index + accessor_offset<Coordinate>(accessor);
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
        template <typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION typename std::enable_if<Coordinate == 1, int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp<StorageInfo>::value;
            constexpr int_t halo = StorageInfo::halo_t::template at<Coordinate>();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_j_block_base;
            return block_base + m_j_block_index + accessor_offset<Coordinate>(accessor);
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
        template <typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION typename std::enable_if<Coordinate == 2, int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            // for ij-caches we simply ignore the block index and always access storage at k = 0
            const int_t block_index =
                (accessor_is_ij_cached<Accessor>::value && m_enable_ij_caches) ? 0 : m_k_block_index;
            return block_index + accessor_offset<Coordinate>(accessor);
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
        template <typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION constexpr typename std::enable_if<(Coordinate > 2), int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            return accessor_offset<Coordinate>(accessor);
        }

        template <typename StorageInfo, typename Accessor, std::size_t... Coordinates>
        GT_FUNCTION int_t compute_offset_impl(Accessor const &accessor, gt_index_sequence<Coordinates...>) const {
            return accumulate(plus_functor(),
                (storage_stride<StorageInfo, Coordinates>() *
                    coordinate_offset<StorageInfo, Coordinates>(accessor))...);
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
        template <typename StorageInfo, typename Accessor>
        GT_FUNCTION int_t compute_offset(Accessor const &accessor) const {
            using sequence_t = make_gt_index_sequence<StorageInfo::layout_t::masked_length>;
            return compute_offset_impl<StorageInfo>(accessor, sequence_t());
        }
    };

    /**
     * @brief Returns the value of the memory at the given address, plus the offset specified by the arg placeholder.
     * @param accessor Accessor passed to the evaluator.
     * @param storage_pointer Pointer to the first element of the specific data field used.
     */
    template <typename IterateDomainArguments>
    template <typename Accessor, typename StoragePointer>
    GT_FUNCTION typename iterate_domain_mic<IterateDomainArguments>::template accessor_return_type<Accessor>::type
    iterate_domain_mic<IterateDomainArguments>::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {
        // getting information about the storage
        using arg_t = typename local_domain_t::template get_arg<typename Accessor::index_t>::type;
        using storage_info_t = typename arg_t::data_store_t::storage_info_t;
        using data_t = typename arg_t::data_store_t::data_t;

        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");

        assert(storage_pointer);
        data_t *RESTRICT real_storage_pointer = static_cast<data_t *>(storage_pointer);

        const int_t pointer_offset = compute_offset<storage_info_t>(accessor);

#ifdef __SSE__
        if (m_prefetch_distance != 0) {
            const int_t prefetch_offset = m_prefetch_distance * storage_stride<storage_info_t, 2>();
            _mm_prefetch(
                reinterpret_cast<const char *>(&real_storage_pointer[pointer_offset + prefetch_offset]), _MM_HINT_T1);
        }
#endif
        return real_storage_pointer[pointer_offset];
    }

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_mic<IterateDomainArguments>> : boost::mpl::true_ {};

} // namespace gridtools
