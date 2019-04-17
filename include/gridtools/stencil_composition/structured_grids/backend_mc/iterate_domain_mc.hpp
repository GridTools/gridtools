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

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/multi_shift.hpp"
#include "../dim.hpp"

namespace gridtools {

    namespace iterate_domain_mc_impl_ {
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

        template <class LocalDomain>
        struct set_base_offset_f {
            LocalDomain const &m_local_domain;
            int_t m_i_block_base;
            int_t m_j_block_base;
            typename LocalDomain::ptr_map_t &m_dst;

            template <class Arg, enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                using sid_t = GT_META_CALL(storage_from_arg, (LocalDomain, Arg));
                using strides_kind_t = GT_META_CALL(sid::strides_kind, sid_t);
                auto length = at_key<strides_kind_t>(m_local_domain.m_total_length_map);
                GT_META_CALL(sid::ptr_diff_type, sid_t) offset = std::lround(length * thread_factor());
                assert(offset == ((long long)length * omp_get_thread_num()) / omp_get_max_threads());
                auto const &strides = at_key<strides_kind_t>(m_local_domain.m_strides_map);
                GT_STATIC_ASSERT(is_storage_info<strides_kind_t>::value, GT_INTERNAL_ERROR);
                sid::shift(offset, sid::get_stride<dim::i>(strides), strides_kind_t::halo_t::template at<0>());
                sid::shift(offset, sid::get_stride<dim::j>(strides), strides_kind_t::halo_t::template at<1>());
                at_key<Arg>(m_dst) += offset;
            }

            template <class Arg, enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                using sid_t = GT_META_CALL(storage_from_arg, (LocalDomain, Arg));
                using strides_kind_t = GT_META_CALL(sid::strides_kind, sid_t);
                auto &ptr = at_key<Arg>(m_dst);
                auto const &strides = at_key<strides_kind_t>(m_local_domain.m_strides_map);
                sid::shift(ptr, sid::get_stride<dim::i>(strides), m_i_block_base);
                sid::shift(ptr, sid::get_stride<dim::j>(strides), m_j_block_base);
            }
        };

#ifdef __SSE__
        template <class T, class PtrDiff, class Strides>
        GT_FORCE_INLINE void do_prefetch(T *ptr, PtrDiff offset, Strides const &strides, int_t dist) {
            if (!dist)
                return;
            sid::shift(offset, sid::get_stride<dim::k>(strides), dist);
            _mm_prefetch(reinterpret_cast<const char *>(ptr + offset), _MM_HINT_T1);
        }

        GT_FORCE_INLINE void do_prefetch(...) {}
#endif

    } // namespace iterate_domain_mc_impl_

    /**
     * @brief Iterate domain class for the MC backend.
     */
    template <class LocalDomain, class IJCachedArgs>
    class iterate_domain_mc {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);

        typename LocalDomain::strides_map_t const &m_strides_map;
        typename LocalDomain::ptr_map_t m_ptr_map;
        int_t m_i_block_index;     /** Local i-index inside block. */
        int_t m_j_block_index;     /** Local j-index inside block. */
        int_t m_k_block_index;     /** Local/global k-index (no blocking along k-axis). */
        int_t m_i_block_base;      /** Global block start index along i-axis. */
        int_t m_j_block_base;      /** Global block start index along j-axis. */
        int_t m_prefetch_distance; /** Prefetching distance along k-axis, zero means no software prefetching. */

      public:
        GT_FORCE_INLINE
        iterate_domain_mc(LocalDomain const &local_domain, int_t i_block_base = 0, int_t j_block_base = 0)
            : m_strides_map(local_domain.m_strides_map), m_ptr_map(local_domain.make_ptr_map()), m_i_block_index(0),
              m_j_block_index(0), m_k_block_index(0), m_i_block_base(i_block_base), m_j_block_base(j_block_base),
              m_prefetch_distance(0) {
            gridtools::for_each_type<typename LocalDomain::esf_args_t>(
                iterate_domain_mc_impl_::set_base_offset_f<LocalDomain>{
                    local_domain, i_block_base, j_block_base, m_ptr_map});
        }

        /** @brief Sets the local block index along the i-axis. */
        GT_FORCE_INLINE void set_i_block_index(int_t i) { m_i_block_index = i; }
        /** @brief Sets the local block index along the j-axis. */
        GT_FORCE_INLINE void set_j_block_index(int_t j) { m_j_block_index = j; }
        /** @brief Sets the local block index along the k-axis. */
        GT_FORCE_INLINE void set_k_block_index(int_t k) { m_k_block_index = k; }

        /** @brief Sets the software prefetching distance along k-axis. Zero means no software prefetching. */
        GT_FORCE_INLINE void set_prefetch_distance(int_t prefetch_distance) { m_prefetch_distance = prefetch_distance; }

        /**
         * @brief Returns the value pointed by an accessor.
         */
        template <class Arg,
            intent Intent,
            class Accessor,
            enable_if_t<!meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
        GT_FORCE_INLINE typename deref_type<Arg, Intent>::type deref(Accessor const &accessor) const {
            using sid_t = GT_META_CALL(storage_from_arg, (LocalDomain, Arg));
            using strides_kind_t = GT_META_CALL(sid::strides_kind, sid_t);
            auto const &strides = at_key<strides_kind_t>(m_strides_map);
            GT_META_CALL(sid::ptr_diff_type, sid_t) ptr_offset{};
            sid::shift(ptr_offset, sid::get_stride<dim::i>(strides), m_i_block_index);
            sid::shift(ptr_offset, sid::get_stride<dim::j>(strides), m_j_block_index);
            sid::shift(ptr_offset, sid::get_stride<dim::k>(strides), m_k_block_index);
            sid::multi_shift(ptr_offset, strides, accessor);
            auto &&ptr = at_key<Arg>(m_ptr_map);
#ifdef __SSE__
            iterate_domain_mc_impl_::do_prefetch(ptr, ptr_offset, strides, m_prefetch_distance);
#endif
            return *(ptr + ptr_offset);
        }

        template <class Arg,
            intent Intent,
            class Accessor,
            enable_if_t<meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
        GT_FORCE_INLINE typename deref_type<Arg, Intent>::type deref(Accessor const &accessor) const {
            using sid_t = GT_META_CALL(storage_from_arg, (LocalDomain, Arg));
            using strides_kind_t = GT_META_CALL(sid::strides_kind, sid_t);
            auto const &strides = at_key<strides_kind_t>(m_strides_map);
            GT_META_CALL(sid::ptr_diff_type, sid_t) ptr_offset{};
            sid::shift(ptr_offset, sid::get_stride<dim::i>(strides), m_i_block_index);
            sid::shift(ptr_offset, sid::get_stride<dim::j>(strides), m_j_block_index);
            sid::multi_shift(ptr_offset, strides, accessor);
            return *(at_key<Arg>(m_ptr_map) + ptr_offset);
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
    };

    template <class LocalDomain, class IJCachedArgs>
    struct is_iterate_domain<iterate_domain_mc<LocalDomain, IJCachedArgs>> : std::true_type {};
} // namespace gridtools
