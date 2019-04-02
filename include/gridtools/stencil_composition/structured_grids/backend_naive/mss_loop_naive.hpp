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

#include "../../backend_naive/basic_token_execution_naive.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../positional_iterate_domain.hpp"
#include "./iterate_domain_naive.hpp"
#include "./run_esf_functor_naive.hpp"

/**@file
 * @brief mss loop implementations for the naive backend
 */
namespace gridtools {
    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(
        backend::naive const &, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        using iterate_domain_arguments_t = iterate_domain_arguments<backend::naive,
            LocalDomain,
            typename RunFunctorArgs::esf_sequence_t,
            std::tuple<>,
            Grid>;
        using iterate_domain_naive_t = iterate_domain_naive<iterate_domain_arguments_t>;
        using iterate_domain_t = typename conditional_t<local_domain_is_stateful<LocalDomain>::value,
            meta::lazy::id<positional_iterate_domain<iterate_domain_naive_t>>,
            meta::lazy::id<iterate_domain_naive_t>>::type;

        iterate_domain_t it_domain(local_domain);

        using extent_t = GT_META_CALL(get_extent_from_loop_intervals, typename RunFunctorArgs::loop_intervals_t);
        using interval_t = GT_META_CALL(meta::first, typename RunFunctorArgs::loop_intervals_t);
        using from_t = GT_META_CALL(meta::first, interval_t);
        it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
            {0, 0, 0},
            {extent_t::iminus::value,
                extent_t::jminus::value,
                static_cast<int_t>(grid.template value_at<from_t>() - grid.k_min())});

        // run the nested ij loop
        const uint_t size_i =
            grid.i_high_bound() - grid.i_low_bound() + 1 + extent_t::iplus::value - extent_t::iminus::value;
        const uint_t size_j =
            grid.j_high_bound() - grid.j_low_bound() + 1 + extent_t::jplus::value - extent_t::jminus::value;
        for (uint_t i = 0; i != size_i; ++i) {
            auto irestore_index = it_domain.index();
            for (uint_t j = 0; j != size_j; ++j) {
                auto jrestore_index = it_domain.index();
                run_functors_on_interval<RunFunctorArgs, run_esf_functor_naive>(it_domain, grid);
                it_domain.set_index(jrestore_index);
                it_domain.increment_j();
            }
            it_domain.set_index(irestore_index);
            it_domain.increment_i();
        }
    }
} // namespace gridtools
