/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * mss_functor.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../meta.hpp"
#include "grid.hpp"
#include "local_domain.hpp"
#include "mss_components.hpp"
#include "mss_components_metafunctions.hpp"
#include "mss_loop.hpp"
#include "run_functor_arguments.hpp"

namespace gridtools {
    /**
     * @brief functor that executes all the functors contained within the mss
     */
    template <typename MssComponentsArray,
        typename Backend,
        typename LocalDomains,
        typename Grid,
        typename ExecutionInfo>
    struct mss_functor {
        GT_STATIC_ASSERT((meta::all_of<is_local_domain, LocalDomains>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponentsArray>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

        LocalDomains const &m_local_domains;
        Grid const &m_grid;
        ExecutionInfo m_execution_info;

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the
         * operations defined on that functor.
         */
        template <typename Index>
        void operator()(Index) const {
            GT_STATIC_ASSERT(Index::value < meta::length<MssComponentsArray>::value, GT_INTERNAL_ERROR);
            using mss_components_t = GT_META_CALL(meta::at, (MssComponentsArray, Index));

            typedef run_functor_arguments<Backend,
                typename mss_components_t::linear_esf_t,
                typename mss_components_t::loop_intervals_t,
                typename mss_components_t::execution_engine_t>
                run_functor_args_t;

            mss_loop<run_functor_args_t>(Backend{}, std::get<Index::value>(m_local_domains), m_grid, m_execution_info);
        }
    };

    template <class MssComponentsArray, class Backend, class LocalDomains, class Grid, class ExecutionInfo>
    void run_mss_functors(
        Backend, LocalDomains const &local_domains, Grid const &grid, ExecutionInfo const &execution_info) {
        for_each<GT_META_CALL(meta::make_indices_for, MssComponentsArray)>(
            mss_functor<MssComponentsArray, Backend, LocalDomains, Grid, ExecutionInfo>{
                local_domains, grid, execution_info});
    }
} // namespace gridtools
