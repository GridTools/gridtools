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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../../storage/common/storage_info.hpp"
#include "../dim.hpp"
#include "../grid.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../sid/concept.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools {
    /**
     * @brief iterate domain class for the naive backend
     */
    template <class LocalDomain>
    class iterate_domain_naive {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);

        using ptr_map_t = typename LocalDomain::ptr_map_t;
        using strides_map_t = typename LocalDomain::strides_map_t;

        template <class Dim>
        struct increment_f {
            iterate_domain_naive *m_self;
            int_t m_offset;

            template <class Arg>
            void operator()() const {
                auto const &stride = sid::get_stride<Dim>(m_self->template strides<Arg>());
                auto &ptr = at_key<Arg>(m_self->m_ptr_map);
                sid::shift(ptr, stride, m_offset);
            }
        };

        template <class Grid>
        struct set_base_offset_f {
            iterate_domain_naive *m_self;
            Grid const &m_grid;

            template <class Arg, enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            void operator()() const {
                auto const &strides = m_self->template strides<Arg>();
                auto &ptr = at_key<Arg>(m_self->m_ptr_map);
                using sid_t = storage_from_arg<LocalDomain, Arg>;
                using strides_kind_t = sid::strides_kind<sid_t>;
                GT_STATIC_ASSERT(is_storage_info<strides_kind_t>::value, GT_INTERNAL_ERROR);
                sid::shift(ptr, sid::get_stride<dim::i>(strides), strides_kind_t::halo_t::template at<dim::i::value>());
                sid::shift(ptr, sid::get_stride<dim::j>(strides), strides_kind_t::halo_t::template at<dim::j::value>());
                sid::shift(ptr, sid::get_stride<dim::k>(strides), -m_grid.k_min());
            }

            template <class Arg, enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            void operator()() const {
                auto &ptr = at_key<Arg>(m_self->m_ptr_map);
                auto const &strides = m_self->template strides<Arg>();
                sid::shift(ptr, sid::get_stride<dim::i>(strides), m_grid.i_low_bound());
                sid::shift(ptr, sid::get_stride<dim::j>(strides), m_grid.j_low_bound());
            }
        };

        ptr_map_t m_ptr_map;
        strides_map_t m_strides_map;
        pos3<int_t> m_pos;

        template <class Arg, class Sid = storage_from_arg<LocalDomain, Arg>, class StridesKind = sid::strides_kind<Sid>>
        decltype(auto) strides() const {
            return at_key<StridesKind>(m_strides_map);
        }

        template <class Dim>
        void increment(int_t offset) {
            for_each_type<typename LocalDomain::esf_args_t>(increment_f<Dim>{this, offset});
        }

      public:
        template <class Grid>
        iterate_domain_naive(LocalDomain const &local_domain, Grid const &grid)
            : m_ptr_map(local_domain.make_ptr_map()),
              m_strides_map(local_domain.m_strides_map), m_pos{(int_t)grid.i_low_bound(),
                                                             (int_t)grid.j_low_bound(),
                                                             (int_t)grid.k_min()} {
            for_each_type<typename LocalDomain::esf_args_t>(set_base_offset_f<Grid>{this, grid});
        }

        void increment_i(int_t offset = 1) {
            m_pos.i += offset;
            increment<dim::i>(offset);
        }
        void increment_j(int_t offset = 1) {
            m_pos.j += offset;
            increment<dim::j>(offset);
        }
        void increment_k(int_t offset = 1) {
            m_pos.k += offset;
            increment<dim::k>(offset);
        }

        void increment_c(int_t offset = 1) { increment<dim::c>(offset); }

        template <class Arg, class Accessor>
        auto deref(Accessor const &accessor) const -> decltype(*at_key<Arg>(m_ptr_map)) {
            auto p = at_key<Arg>(m_ptr_map);
            sid::multi_shift(p, strides<Arg>(), accessor);
            return *p;
        }

        int_t i() const { return m_pos.i; }
        int_t j() const { return m_pos.j; }
        int_t k() const { return m_pos.k; }
    };

    template <class LocalDomain>
    struct is_iterate_domain<iterate_domain_naive<LocalDomain>> : std::true_type {};
} // namespace gridtools
