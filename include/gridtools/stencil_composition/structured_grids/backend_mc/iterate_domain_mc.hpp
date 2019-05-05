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

#include <cmath>

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../../positional.hpp"
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
            typename LocalDomain::ptr_t &m_dst;

            template <class Arg, class Dim>
            GT_FORCE_INLINE auto stride() const GT_AUTO_RETURN((sid::get_stride<Arg, Dim>(m_local_domain.m_strides)));

            template <class Arg, enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                using storage_info_t = typename Arg::data_store_t::storage_info_t;
                GT_STATIC_ASSERT(is_storage_info<storage_info_t>::value, GT_INTERNAL_ERROR);
                auto length = at_key<storage_info_t>(m_local_domain.m_total_length_map);
                int_t offset = std::lround(length * thread_factor());
                assert(offset == ((long long)length * omp_get_thread_num()) / omp_get_max_threads());

                sid::shift(offset, stride<Arg, dim::i>(), storage_info_t::halo_t::template at<0>());
                sid::shift(offset, stride<Arg, dim::j>(), storage_info_t::halo_t::template at<1>());
                at_key<Arg>(m_dst) += offset;
            }

            template <class Arg, enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                auto &ptr = at_key<Arg>(m_dst);
                sid::shift(ptr, stride<Arg, dim::i>(), m_i_block_base);
                sid::shift(ptr, stride<Arg, dim::j>(), m_j_block_base);
            }
        };
    } // namespace iterate_domain_mc_impl_

    /**
     * @brief Iterate domain class for the MC backend.
     */
    template <class LocalDomain, class IJCachedArgs>
    class iterate_domain_mc {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);

        using strides_t = typename LocalDomain::strides_t;
        using ptr_t = typename LocalDomain::ptr_t;

        strides_t const &m_strides;
        ptr_t m_ptr;

        int_t m_i_block_index; /** Local i-index inside block. */
        int_t m_j_block_index; /** Local j-index inside block. */
        int_t m_k_block_index; /** Local/global k-index (no blocking along k-axis). */

        template <class Arg, class Dim>
        GT_FORCE_INLINE auto stride() const GT_AUTO_RETURN((sid::get_stride<Arg, Dim>(m_strides)));

        GT_FORCE_INLINE positional pos() const {
            positional res = at_key<positional>(m_ptr);
            sid::shift(res, stride<positional, dim::i>(), m_i_block_index);
            sid::shift(res, stride<positional, dim::j>(), m_j_block_index);
            sid::shift(res, stride<positional, dim::k>(), m_k_block_index);
            return res;
        }

      public:
        GT_FORCE_INLINE
        iterate_domain_mc(LocalDomain const &local_domain, int_t i_block_base = 0, int_t j_block_base = 0)
            : m_strides(local_domain.m_strides), m_ptr(local_domain.m_ptr_holder()), m_i_block_index(0),
              m_j_block_index(0), m_k_block_index(0) {
            for_each_type<typename LocalDomain::esf_args_t>(iterate_domain_mc_impl_::set_base_offset_f<LocalDomain>{
                local_domain, i_block_base, j_block_base, m_ptr});
        }

        /** @brief Sets the local block index along the i-axis. */
        GT_FORCE_INLINE void set_i_block_index(int_t i) { m_i_block_index = i; }
        /** @brief Sets the local block index along the j-axis. */
        GT_FORCE_INLINE void set_j_block_index(int_t j) { m_j_block_index = j; }
        /** @brief Sets the local block index along the k-axis. */
        GT_FORCE_INLINE void set_k_block_index(int_t k) { m_k_block_index = k; }

        /**
         * @brief Returns the value pointed by an accessor.
         */
        template <class Arg, class Accessor, enable_if_t<!meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
        GT_FORCE_INLINE auto deref(Accessor const &accessor) const -> decltype(*at_key<Arg>(m_ptr)) {
            auto ptr = at_key<Arg>(m_ptr);
            sid::shift(ptr, stride<Arg, dim::i>(), m_i_block_index);
            sid::shift(ptr, stride<Arg, dim::j>(), m_j_block_index);
            sid::shift(ptr, stride<Arg, dim::k>(), m_k_block_index);
            sid::multi_shift<Arg>(ptr, m_strides, accessor);
            return *ptr;
        }

        template <class Arg, class Accessor, enable_if_t<meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
        GT_FORCE_INLINE auto deref(Accessor const &accessor) const -> decltype(*at_key<Arg>(m_ptr)) {
            auto ptr = at_key<Arg>(m_ptr);
            sid::shift(ptr, stride<Arg, dim::i>(), m_i_block_index);
            sid::shift(ptr, stride<Arg, dim::j>(), m_j_block_index);
            sid::multi_shift<Arg>(ptr, m_strides, accessor);
            return *ptr;
        }

        /** @brief Global i-index. */
        GT_FORCE_INLINE
        int_t i() const { return pos().i; }

        /** @brief Global j-index. */
        GT_FORCE_INLINE
        int_t j() const { return pos().j; }

        /** @brief Global k-index. */
        GT_FORCE_INLINE
        int_t k() const { return pos().k; }
    };

    template <class LocalDomain, class IJCachedArgs>
    struct is_iterate_domain<iterate_domain_mc<LocalDomain, IJCachedArgs>> : std::true_type {};
} // namespace gridtools
