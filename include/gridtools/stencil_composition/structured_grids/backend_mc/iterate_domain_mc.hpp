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

        template <class Offset>
        struct k_shift_f {
            ptr_t &m_ptr;
            strides_t const &m_strides;
            Offset m_offset;

            template <class Arg, enable_if_t<!meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                sid::shift(at_key<Arg>(m_ptr), sid::get_stride<Arg, dim::k>(m_strides), m_offset);
            }
            template <class Arg, enable_if_t<meta::st_contains<IJCachedArgs, Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {}
        };

      public:
        GT_FORCE_INLINE
        iterate_domain_mc(LocalDomain const &local_domain, int_t i_block_base = 0, int_t j_block_base = 0)
            : m_strides(local_domain.m_strides), m_ptr(local_domain.m_ptr_holder()) {
            for_each_type<GT_META_CALL(get_keys, ptr_t)>(iterate_domain_mc_impl_::set_base_offset_f<LocalDomain>{
                local_domain, i_block_base, j_block_base, m_ptr});
        }

        template <class Offset>
        GT_FORCE_INLINE void k_shift(ptr_t &ptr, Offset offset) const {
            for_each_type<GT_META_CALL(get_keys, ptr_t)>(k_shift_f<Offset>{ptr, m_strides, offset});
        }

        GT_FORCE_INLINE ptr_t &ptr() { return m_ptr; }
        GT_FORCE_INLINE strides_t const &strides() const { return m_strides; }
    };

    template <class LocalDomain, class IJCachedArgs>
    struct is_iterate_domain<iterate_domain_mc<LocalDomain, IJCachedArgs>> : std::true_type {};
} // namespace gridtools
