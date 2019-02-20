/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include <cmath>
#include <functional>

#include <boost/fusion/functional/invocation/invoke.hpp>

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/gt_assert.hpp"
#include "../../../meta.hpp"
#include "../../accessor_base.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../global_accessor.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../offset_computation.hpp"

namespace gridtools {

    namespace _impl {
        /**
         * @brief Per-thread global value of omp_get_thread_num() / omp_get_max_threads().
         */
        inline float thread_factor() {
#if !defined(__APPLE_CC__) || __APPLE_CC__ > 8000
            thread_local static
#endif
                const float value = (float)omp_get_thread_num() / omp_get_max_threads();
            return value;
        }

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

            template <class ArgDataPtrPair>
            GT_FUNCTION void operator()(ArgDataPtrPair const &) const {
                using arg_t = typename ArgDataPtrPair::first_type;
                constexpr auto arg_index = meta::st_position<typename LocalDomain::esf_args, arg_t>::value;

                get<arg_index>(m_data_ptr_offsets) = fields_offset<arg_t>(); // non-zero only for tmps.
            }

          private:
            template <typename Arg>
            GT_FUNCTION enable_if_t<is_tmp_arg<Arg>::value, int_t> fields_offset() const {
                using storage_info_ptr_t = typename Arg::data_store_t::storage_info_t const *;
                constexpr auto storage_info_index =
                    meta::st_position<typename LocalDomain::storage_info_ptr_list, storage_info_ptr_t>::value;
                storage_info_ptr_t storage_info =
                    boost::fusion::at_c<storage_info_index>(m_local_domain.m_local_storage_info_ptrs);

                int_t offset = std::lround(storage_info->padded_total_length() * thread_factor());
                assert(offset ==
                       ((long long)storage_info->padded_total_length() * omp_get_thread_num()) / omp_get_max_threads());
                return offset;
            }

            template <typename Arg>
            GT_FUNCTION enable_if_t<!is_tmp_arg<Arg>::value, int_t> fields_offset() const {
                return 0;
            }
        };

    } // namespace _impl

    /**
     * @brief Iterate domain class for the MC backend.
     */
    template <typename IterateDomainArguments>
    class iterate_domain_mc {
        GT_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using backend_traits_t = backend_traits_from_id<target::mc>;

        using esf_sequence_t = typename IterateDomainArguments::esf_sequence_t;
        using cache_sequence_t = typename IterateDomainArguments::cache_sequence_t;

        /* meta function to get storage info index in local domain */
        template <typename StorageInfo>
        using local_domain_storage_index =
            meta::st_position<typename local_domain_t::storage_info_ptr_list, const StorageInfo *>;

        /* meta function to check if a storage info belongs to a temporary field */
        template <typename StorageInfo>
        using storage_is_tmp =
            meta::st_contains<typename local_domain_t::tmp_storage_info_ptr_list, StorageInfo const *>;

        using ij_cache_args_t = GT_META_CALL(ij_cache_args, typename IterateDomainArguments::cache_sequence_t);

      public:
        using storage_info_ptrs_t = typename local_domain_t::storage_info_ptr_fusion_list;

        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = boost::mpl::size<storage_info_ptrs_t>::value;

        using strides_cached_t = strides_cached<n_meta_storages - 1, storage_info_ptrs_t>;
        using array_index_t = array<int_t, n_meta_storages>;
        // *************** end of type definitions **************

      private:
        // *********************** members **********************
        local_domain_t const &local_domain;
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
            index_getter(iterate_domain_mc const &it_domain, array_index_t &index_array)
                : m_it_domain(it_domain), m_index_array(index_array) {}

            template <class StorageInfoIndex>
            void operator()(StorageInfoIndex const &) const {
                using storage_info_ptrref_t =
                    typename boost::fusion::result_of::at<typename local_domain_t::storage_info_ptr_fusion_list,
                        StorageInfoIndex>::type;
                using storage_info_t = typename std::remove_const<typename std::remove_pointer<
                    typename std::remove_reference<storage_info_ptrref_t>::type>::type>::type;
                static constexpr bool is_ij_cached = false;
                m_index_array[StorageInfoIndex::value] =
                    m_it_domain.compute_offset<is_ij_cached, storage_info_t>(accessor_base<storage_info_t::ndims>());
            }

          private:
            iterate_domain_mc const &m_it_domain;
            array<int_t, n_meta_storages> &m_index_array;
        };

        using data_ptr_offsets_t =
            array<int, boost::fusion::result_of::size<decltype(local_domain.m_local_data_ptrs)>::value>;
        data_ptr_offsets_t m_data_ptr_offsets;

      public:
        GT_FUNCTION
        iterate_domain_mc(local_domain_t const &local_domain)
            : local_domain(local_domain), m_i_block_index(0), m_j_block_index(0), m_k_block_index(0), m_i_block_base(0),
              m_j_block_base(0), m_prefetch_distance(0), m_enable_ij_caches(false) {
            // assign stride pointers
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                assign_strides<backend_traits_t, strides_cached_t, local_domain_t>{m_strides});
            boost::fusion::for_each(local_domain.m_local_data_ptrs,
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
            for_each<GT_META_CALL(meta::make_indices_c, n_meta_storages)>(index_getter(*this, index_array));
            return index_array;
        }

        /** @brief Sets the software prefetching distance along k-axis. Zero means no software prefetching. */
        GT_FUNCTION void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        /** @brief Enables ij-caches. */
        GT_FUNCTION void enable_ij_caches() { m_enable_ij_caches = true; }

        /**
         * @brief Method called in the apply methods of the functors.
         * Specialization for the global accessors placeholders.
         */
        template <class Arg, intent Intent, uint_t I>
        GT_FORCE_INLINE typename Arg::data_store_t::data_t deref(global_accessor<I> const &) const {
            return *boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs);
        }

        /**
         * @brief Method called in the apply methods of the functors.
         * Specialization for the global accessors placeholders with arguments.
         */
        template <class Arg, intent Intent, class Acc, class... Args>
        GT_FORCE_INLINE auto deref(global_accessor_with_arguments<Acc, Args...> const &acc) const
            GT_AUTO_RETURN(boost::fusion::invoke(
                std::cref(*boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs)), acc.get_arguments()));

        /**
         * @brief Returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression).
         */
        template <class Arg,
            intent Intent,
            class Accessor,
            enable_if_t<is_accessor<Accessor>::value && !is_global_accessor<Accessor>::value, int> = 0>
        GT_FORCE_INLINE typename deref_type<Arg, Intent>::type deref(Accessor const &accessor) const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;
            using data_t = typename Arg::data_store_t::data_t;

            static constexpr auto storage_index = local_domain_storage_index<storage_info_t>::value;
            static constexpr auto arg_index = meta::st_position<typename local_domain_t::esf_args, Arg>::value;

            const storage_info_t *storage_info =
                boost::fusion::at_c<storage_index>(local_domain.m_local_storage_info_ptrs);

            auto ptr = boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs) + m_data_ptr_offsets[arg_index];

            int_t pointer_offset =
                compute_offset<meta::st_contains<ij_cache_args_t, Arg>::value, storage_info_t>(accessor);

#ifdef __SSE__
            if (m_prefetch_distance != 0) {
                int_t prefetch_offset = m_prefetch_distance * storage_stride<storage_info_t, 2>();
                _mm_prefetch(reinterpret_cast<const char *>(&ptr[pointer_offset + prefetch_offset]), _MM_HINT_T1);
            }
#endif
            return ptr[pointer_offset];
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
            static constexpr auto storage_index = local_domain_storage_index<StorageInfo>::value;
            auto const &strides = m_strides.template get<storage_index>();
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
        template <bool, typename StorageInfo, int_t Coordinate, typename Accessor>
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
        template <bool, typename StorageInfo, int_t Coordinate, typename Accessor>
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
        template <bool IsIjCached, typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION typename std::enable_if<Coordinate == 2, int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            // for ij-caches we simply ignore the block index and always access storage at k = 0
            const int_t block_index = IsIjCached && m_enable_ij_caches ? 0 : m_k_block_index;
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
        template <bool, typename StorageInfo, int_t Coordinate, typename Accessor>
        GT_FUNCTION constexpr typename std::enable_if<(Coordinate > 2), int_t>::type coordinate_offset(
            Accessor const &accessor) const {
            return accessor_offset<Coordinate>(accessor);
        }

        template <bool IsIjCached, typename StorageInfo, typename Accessor, std::size_t... Coordinates>
        GT_FUNCTION int_t compute_offset_impl(Accessor const &accessor, meta::index_sequence<Coordinates...>) const {
            return accumulate(plus_functor(),
                (storage_stride<StorageInfo, Coordinates>() *
                    coordinate_offset<IsIjCached, StorageInfo, Coordinates>(accessor))...);
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
        template <bool IsIjCached, typename StorageInfo, typename Accessor>
        GT_FUNCTION int_t compute_offset(Accessor const &accessor) const {
            using sequence_t = meta::make_index_sequence<StorageInfo::layout_t::masked_length>;
            return compute_offset_impl<IsIjCached, StorageInfo>(accessor, sequence_t());
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_mc<IterateDomainArguments>> : std::true_type {};

} // namespace gridtools
