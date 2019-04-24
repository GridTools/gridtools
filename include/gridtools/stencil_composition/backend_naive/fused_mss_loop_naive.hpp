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

#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../grid.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "iterate_domain_naive.hpp"

namespace gridtools {
    namespace naive_impl_ {
        template <class LocalDomain, class Grid>
        struct stage_executor_f {
            LocalDomain const &m_local_domain;
            Grid const &m_grid;

            template <template <class...> class L, class From, class To, class Stage>
            void operator()(L<From, To, Stage>) const {
                using extent_t = typename Stage::extent_t;
                int_t iminus = extent_t::iminus::value;
                int_t iplus = extent_t::iplus::value;
                int_t jminus = extent_t::jminus::value;
                int_t jplus = extent_t::jplus::value;
                int_t k_from = m_grid.template value_at<From>();
                int_t k_to = m_grid.template value_at<To>();
                int_t i_count = m_grid.i_high_bound() - m_grid.i_low_bound() + 1 + iplus - iminus;
                int_t j_count = m_grid.j_high_bound() - m_grid.j_low_bound() + 1 + jplus - jminus;
                int_t k_count = 1 + std::abs(k_to - k_from);
                int_t k_step = k_to >= k_from ? 1 : -1;

                iterate_domain_naive<LocalDomain> it_domain(m_local_domain, m_grid);

                it_domain.increment_i(iminus);
                it_domain.increment_j(jminus);
                it_domain.increment_k(k_from);

                for (int_t i = 0; i != i_count; ++i) {
                    for (int_t j = 0; j != j_count; ++j) {
                        for (int_t k = 0; k != k_count; ++k) {
                            Stage::exec(it_domain);
                            it_domain.increment_k(k_step);
                        }
                        it_domain.increment_k(-k_count * k_step);
                        it_domain.increment_j();
                    }
                    it_domain.increment_j(-j_count);
                    it_domain.increment_i();
                }
            }
        };

        template <class LoopInterval,
            class From = GT_META_CALL(meta::first, LoopInterval),
            class To = GT_META_CALL(meta::second, LoopInterval),
            class StageGroups = GT_META_CALL(meta::third, LoopInterval),
            class Stages = GT_META_CALL(meta::flatten, StageGroups)>
        GT_META_DEFINE_ALIAS(
            split_loop_interval, meta::transform, (meta::curry<meta::list, From, To>::template apply, Stages));

        template <class LoopIntervals>
        GT_META_DEFINE_ALIAS(
            split_loop_intervals, meta::flatten, (GT_META_CALL(meta::transform, (split_loop_interval, LoopIntervals))));

        template <class Grid>
        struct mss_executor_f {
            Grid const &m_grid;
            template <class MssComponents, class LocalDomain>
            void operator()(MssComponents, LocalDomain const &local_domain) const {
                GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
                using loop_intervals_t = GT_META_CALL(split_loop_intervals, typename MssComponents::loop_intervals_t);
                for_each<loop_intervals_t>(stage_executor_f<LocalDomain, Grid>{local_domain, m_grid});
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
