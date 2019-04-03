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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include <cmath>
#include <functional>

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/gt_assert.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../accessor_base.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/multi_shift.hpp"
#include "../dim.hpp"

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
         */
        template <typename LocalDomain>
        struct set_offset_for_temporaries_f {
            LocalDomain const &m_local_domain;
            typename LocalDomain::ptr_map_t &m_dst;

            template <typename Arg>
            GT_FORCE_INLINE void operator()() const {
                auto length = at_key<typename Arg::data_store_t::storage_info_t>(m_local_domain.m_total_length_map);
                int_t offset = std::lround(length * thread_factor());
                assert(offset == ((long long)length * omp_get_thread_num()) / omp_get_max_threads());
                at_key<Arg>(m_dst) += offset;
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

        using esf_sequence_t = typename IterateDomainArguments::esf_sequence_t;
        using cache_sequence_t = typename IterateDomainArguments::cache_sequence_t;

        /* meta function to get storage info index in local domain */
        template <typename StorageInfo>
        using local_domain_storage_index = meta::st_position<typename local_domain_t::strides_kinds_t, StorageInfo>;

        /* meta function to check if a storage info belongs to a temporary field */
        template <typename StorageInfo>
        GT_META_DEFINE_ALIAS(
            storage_is_tmp, meta::st_contains, (typename local_domain_t::tmp_strides_kinds_t, StorageInfo));

        using ij_cache_args_t = GT_META_CALL(ij_cache_args, typename IterateDomainArguments::cache_sequence_t);

      public:
        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = meta::length<typename local_domain_t::strides_kinds_t>::value;

        using array_index_t = array<int_t, n_meta_storages>;
        // *************** end of type definitions **************

      private:
        // *********************** members **********************
        local_domain_t const &local_domain;
        int_t m_i_block_index;     /** Local i-index inside block. */
        int_t m_j_block_index;     /** Local j-index inside block. */
        int_t m_k_block_index;     /** Local/global k-index (no blocking along k-axis). */
        int_t m_i_block_base;      /** Global block start index along i-axis. */
        int_t m_j_block_base;      /** Global block start index along j-axis. */
        int_t m_prefetch_distance; /** Prefetching distance along k-axis, zero means no software prefetching. */
        bool m_enable_ij_caches;   /** Enables ij-caching. */
        typename local_domain_t::ptr_map_t m_ptr_map;
        // ******************* end of members *******************

        // helper class for index array generation, only needed for the index() function
        struct index_getter {
            index_getter(iterate_domain_mc const &it_domain, array_index_t &index_array)
                : m_it_domain(it_domain), m_index_array(index_array) {}

            template <class StorageInfoIndex>
            void operator()(StorageInfoIndex const &) const {
                using storage_info_t =
                    GT_META_CALL(meta::at, (typename local_domain_t::strides_kinds_t, StorageInfoIndex));
                static constexpr bool is_ij_cached = false;
                m_index_array[StorageInfoIndex::value] =
                    m_it_domain.compute_offset<is_ij_cached, storage_info_t>(accessor_base<storage_info_t::ndims>());
            }

          private:
            iterate_domain_mc const &m_it_domain;
            array<int_t, n_meta_storages> &m_index_array;
        };

      public:
        GT_FORCE_INLINE
        iterate_domain_mc(local_domain_t const &local_domain)
            : local_domain(local_domain), m_i_block_index(0), m_j_block_index(0), m_k_block_index(0), m_i_block_base(0),
              m_j_block_base(0), m_prefetch_distance(0), m_enable_ij_caches(false),
              m_ptr_map(local_domain.make_ptr_map()) {
            using tmp_args_t = GT_META_CALL(meta::filter, (is_tmp_arg, typename local_domain_t::esf_args_t));
            gridtools::for_each_type<tmp_args_t>(
                _impl::set_offset_for_temporaries_f<local_domain_t>{local_domain, m_ptr_map});
        }

        /** @brief Sets the block start indices. */
        GT_FORCE_INLINE void set_block_base(int_t i_block_base, int_t j_block_base) {
            m_i_block_base = i_block_base;
            m_j_block_base = j_block_base;
        }

        /** @brief Sets the local block index along the i-axis. */
        GT_FORCE_INLINE void set_i_block_index(int_t i) { m_i_block_index = i; }
        /** @brief Sets the local block index along the j-axis. */
        GT_FORCE_INLINE void set_j_block_index(int_t j) { m_j_block_index = j; }
        /** @brief Sets the local block index along the k-axis. */
        GT_FORCE_INLINE void set_k_block_index(int_t k) { m_k_block_index = k; }

        /** @brief Returns the current data index at offset (0, 0, 0) per meta storage. */
        GT_FORCE_INLINE array_index_t index() const {
            array_index_t index_array;
            for_each<GT_META_CALL(meta::make_indices_c, n_meta_storages)>(index_getter(*this, index_array));
            return index_array;
        }

        /** @brief Sets the software prefetching distance along k-axis. Zero means no software prefetching. */
        GT_FORCE_INLINE void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        /** @brief Enables ij-caches. */
        GT_FORCE_INLINE void enable_ij_caches() { m_enable_ij_caches = true; }

        /**
         * @brief Returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression).
         */
        template <class Arg, intent Intent, class Accessor>
        GT_FORCE_INLINE typename deref_type<Arg, Intent>::type deref(Accessor const &accessor) const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            auto ptr = at_key<Arg>(m_ptr_map);

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
        GT_FORCE_INLINE
        int_t i() const { return m_i_block_base + m_i_block_index; }

        /** @brief Global j-index. */
        GT_FORCE_INLINE
        int_t j() const { return m_j_block_base + m_j_block_index; }

        /** @brief Global k-index. */
        GT_FORCE_INLINE
        int_t k() const { return m_k_block_index; }

      private:
        /**
         * @brief Returns stride for a storage along the given axis.
         *
         * @tparam StorageInfo Storage info for which the strides should be returned.
         * @tparam Coordinate Axis/coordinate along which the stride is needed.
         */
        template <typename StorageInfo, int_t Coordinate>
        GT_FORCE_INLINE int_t storage_stride() const {
            return sid::get_stride<integral_constant<int, Coordinate>>(at_key<StorageInfo>(local_domain.m_strides_map));
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
        GT_FORCE_INLINE enable_if_t<Coordinate == 0, int_t> coordinate_offset(Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp<StorageInfo>::value;
            constexpr int_t halo = StorageInfo::halo_t::template at<Coordinate>();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_i_block_base;
            return block_base + m_i_block_index + host_device::at_key<dim::i>(accessor);
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
        GT_FORCE_INLINE enable_if_t<Coordinate == 1, int_t> coordinate_offset(Accessor const &accessor) const {
            constexpr bool is_tmp = storage_is_tmp<StorageInfo>::value;
            constexpr int_t halo = StorageInfo::halo_t::template at<Coordinate>();

            // for temporaries the first element starts after the halo, for other storages we use the block base index
            const int_t block_base = is_tmp ? halo : m_j_block_base;
            return block_base + m_j_block_index + host_device::at_key<dim::j>(accessor);
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
        GT_FORCE_INLINE enable_if_t<Coordinate == 2, int_t> coordinate_offset(Accessor const &accessor) const {
            // for ij-caches we simply ignore the block index and always access storage at k = 0
            const int_t block_index = IsIjCached && m_enable_ij_caches ? 0 : m_k_block_index;
            return block_index + host_device::at_key<dim::k>(accessor);
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
        GT_FORCE_INLINE constexpr enable_if_t<(Coordinate > 2), int_t> coordinate_offset(
            Accessor const &accessor) const {
            return host_device::at_key<integral_constant<int, Coordinate>>(accessor);
        }

        template <bool IsIjCached, typename StorageInfo, typename Accessor, std::size_t... Coordinates>
        GT_FORCE_INLINE int_t compute_offset_impl(
            Accessor const &accessor, meta::index_sequence<Coordinates...>) const {
            return accumulate(plus_functor(),
                0,
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
        GT_FORCE_INLINE int_t compute_offset(Accessor const &accessor) const {
            using sequence_t = meta::make_index_sequence<tuple_util::size<Accessor>::value>;
            return compute_offset_impl<IsIjCached, StorageInfo>(accessor, sequence_t());
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_mc<IterateDomainArguments>> : std::true_type {};

} // namespace gridtools
