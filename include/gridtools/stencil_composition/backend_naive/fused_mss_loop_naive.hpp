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

#include <cstdlib>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../storage/common/storage_info.hpp"
#include "../arg.hpp"
#include "../dim.hpp"
#include "../grid.hpp"
#include "../local_domain.hpp"
#include "../sid/concept.hpp"
#include "../sid/loop.hpp"

namespace gridtools {
    namespace naive_impl_ {

        template <class Ptr, class Strides, class Grid>
        struct correct_ptr_f {
            Ptr &m_ptr;
            Strides const &m_strides;
            Grid const &m_grid;

            template <class Arg, class Dim>
            auto stride() const GT_AUTO_RETURN((sid::get_stride<Arg, dim::i>(m_strides)));

            template <class Arg, enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            void operator()() const {
                auto &ptr = at_key<Arg>(m_ptr);
                using storage_info_t = typename Arg::data_store_t::storage_info_t;
                GT_STATIC_ASSERT(is_storage_info<storage_info_t>::value, GT_INTERNAL_ERROR);
                sid::shift(ptr, stride<Arg, dim::i>(), storage_info_t::halo_t::template at<dim::i::value>());
                sid::shift(ptr, stride<Arg, dim::j>(), storage_info_t::halo_t::template at<dim::j::value>());
                sid::shift(ptr, stride<Arg, dim::k>(), -m_grid.k_min());
            }

            template <class Arg, enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            void operator()() const {
                auto &ptr = at_key<Arg>(m_ptr);
                sid::shift(ptr, stride<Arg, dim::i>(), m_grid.i_low_bound());
                sid::shift(ptr, stride<Arg, dim::j>(), m_grid.j_low_bound());
            }
        };

        // execute a stage with the given local domain on the given computation area defined by grid
        template <class Ptr, class Strides, class Grid>
        struct stage_executor_f {
            Ptr m_ptr;
            Strides const &m_strides;
            Grid const &m_grid;

            template <template <class...> class L, class From, class To, class Stage>
            void operator()(L<From, To, Stage>) const {
                using extent_t = typename Stage::extent_t;

                int_t k_from = m_grid.template value_at<From>();
                int_t k_to = m_grid.template value_at<To>();

                Ptr ptr = m_ptr;

                sid::shift(ptr, sid::get_stride<dim::i>(m_strides), extent_t::iminus::value);
                sid::shift(ptr, sid::get_stride<dim::j>(m_strides), extent_t::jminus::value);
                sid::shift(ptr, sid::get_stride<dim::k>(m_strides), k_from);

                auto i_loop = sid::make_loop<dim::i>(int_t(m_grid.i_high_bound() - m_grid.i_low_bound() + 1 +
                                                           extent_t::iplus::value - extent_t::iminus::value));
                auto j_loop = sid::make_loop<dim::j>(int_t(m_grid.j_high_bound() - m_grid.j_low_bound() + 1 +
                                                           extent_t::jplus::value - extent_t::jminus::value));
                auto k_loop = sid::make_loop<dim::k>(1 + std::abs(k_to - k_from), k_to >= k_from ? 1 : -1);

                i_loop(j_loop(k_loop(Stage{})))(ptr, m_strides);
            }
        };

        template <class Ptr, class Strides, class Grid>
        stage_executor_f<Ptr, Strides, Grid> stage_executor(Ptr ptr, Strides const &strides, Grid const &grid) {
            for_each_type<GT_META_CALL(get_keys, Ptr)>(correct_ptr_f<Ptr, Strides, Grid>{ptr, strides, grid});
            return {ptr, strides, grid};
        }

        // split the loop interval into the list of triples meta::list<From, To, Stage>
        template <class LoopInterval,
            class From = GT_META_CALL(meta::first, LoopInterval),
            class To = GT_META_CALL(meta::second, LoopInterval),
            class StageGroups = GT_META_CALL(meta::third, LoopInterval),
            class Stages = GT_META_CALL(meta::flatten, StageGroups)>
        GT_META_DEFINE_ALIAS(
            split_loop_interval, meta::transform, (meta::curry<meta::list, From, To>::template apply, Stages));

        // split the list of loop intervals into the list of triples meta::list<From, To, Stage>
        template <class LoopIntervals>
        GT_META_DEFINE_ALIAS(
            split_loop_intervals, meta::flatten, (GT_META_CALL(meta::transform, (split_loop_interval, LoopIntervals))));

        // execute stages in mss
        template <class Grid>
        struct mss_executor_f {
            Grid const &m_grid;
            template <class MssComponents, class LocalDomain>
            void operator()(MssComponents, LocalDomain const &local_domain) const {
                GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
                using loop_intervals_t = GT_META_CALL(split_loop_intervals, typename MssComponents::loop_intervals_t);
                for_each<loop_intervals_t>(stage_executor(local_domain.m_ptr_holder(), local_domain.m_strides, m_grid));
            }
        };
    } // namespace naive_impl_

    /**
     * @brief loops over all blocks and execute sequentially all mss functors
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomains, class Grid>
    void fused_mss_loop(backend::naive, LocalDomains const &local_domains, Grid const &grid) {
        tuple_util::for_each(naive_impl_::mss_executor_f<Grid>{grid}, MssComponents{}, local_domains);
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    std::true_type mss_fuse_esfs(backend::naive);
} // namespace gridtools
