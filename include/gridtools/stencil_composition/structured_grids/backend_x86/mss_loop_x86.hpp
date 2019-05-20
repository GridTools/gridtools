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

#include "../../backend_x86/basic_token_execution_x86.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../positional_iterate_domain.hpp"
#include "./iterate_domain_x86.hpp"
#include "./run_esf_functor_x86.hpp"

/**@file
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {
    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(backend::x86 const &backend_target,
        LocalDomain const &local_domain,
        Grid const &grid,
        const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        using iterate_domain_arguments_t =
            iterate_domain_arguments<backend::x86, LocalDomain, typename RunFunctorArgs::esf_sequence_t>;
        using iterate_domain_x86_t = iterate_domain_x86<iterate_domain_arguments_t>;
        using iterate_domain_t = typename conditional_t<local_domain_is_stateful<LocalDomain>::value,
            meta::lazy::id<positional_iterate_domain<iterate_domain_x86_t>>,
            meta::lazy::id<iterate_domain_x86_t>>::type;
        iterate_domain_t it_domain(local_domain);

        using extent_t = get_extent_from_loop_intervals<typename RunFunctorArgs::loop_intervals_t>;
        using interval_t = meta::first<typename RunFunctorArgs::loop_intervals_t>;
        using from_t = meta::first<interval_t>;
        it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
            {execution_info.bi, execution_info.bj, 0},
            {extent_t::iminus::value,
                extent_t::jminus::value,
                static_cast<int_t>(grid.template value_at<from_t>() - grid.k_min())});

        auto block_size_f = [](uint_t total, uint_t block_size, uint_t block_no) {
            auto n = (total + block_size - 1) / block_size;
            return block_no == n - 1 ? total - block_no * block_size : block_size;
        };
        auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
        auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;
        const uint_t size_i = block_size_f(total_i, block_i_size(backend_target), execution_info.bi) +
                              extent_t::iplus::value - extent_t::iminus::value;
        const uint_t size_j = block_size_f(total_j, block_j_size(backend_target), execution_info.bj) +
                              extent_t::jplus::value - extent_t::jminus::value;

        // run the nested ij loop
        for (uint_t i = 0; i != size_i; ++i) {
            auto irestore_index = it_domain.index();
            for (uint_t j = 0; j != size_j; ++j) {
                auto jrestore_index = it_domain.index();
                run_functors_on_interval<RunFunctorArgs, run_esf_functor_x86>(it_domain, grid);
                it_domain.set_index(jrestore_index);
                it_domain.increment_j();
            }
            it_domain.set_index(irestore_index);
            it_domain.increment_i();
        }
    }
} // namespace gridtools
