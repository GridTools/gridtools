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

#include <boost/fusion/include/at.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/size.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../meta/type_traits.hpp"
#include "./backend_traits_fwd.hpp"
#include "./grid.hpp"
#include "./local_domain.hpp"
#include "./mss_components.hpp"
#include "./mss_components_metafunctions.hpp"
#include "./mss_loop.hpp"
#include "./run_functor_arguments.hpp"

namespace gridtools {
    /**
     * @brief functor that executes all the functors contained within the mss
     */
    template <typename MssComponentsArray,
        typename BackendTarget,
        typename LocalDomains,
        typename Grid,
        typename ExecutionInfo>
    struct mss_functor {
      private:
        GT_STATIC_ASSERT((is_sequence_of<LocalDomains, is_local_domain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_sequence_of<MssComponentsArray, is_mss_components>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        BackendTarget const &m_backend_target;
        LocalDomains const &m_local_domains;
        const Grid &m_grid;
        const ExecutionInfo m_execution_info;

      public:
        mss_functor(BackendTarget const &backend_target,
            LocalDomains const &local_domains,
            const Grid &grid,
            const ExecutionInfo &execution_info)
            : m_backend_target(backend_target), m_local_domains(local_domains), m_grid(grid),
              m_execution_info(execution_info) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the
         * operations defined on that functor.
         */
        template <typename Index>
        GT_FUNCTION_HOST void operator()(Index) const {
            GT_STATIC_ASSERT((Index::value < boost::mpl::size<MssComponentsArray>::value), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::at<MssComponentsArray, Index>::type mss_components_t;

            auto const &local_domain = boost::fusion::at<Index>(m_local_domains);

            // wrapping all the template arguments in a single container

            typedef run_functor_arguments<BackendTarget,
                typename mss_components_t::linear_esf_t,
                typename mss_components_t::loop_intervals_t,
                decay_t<decltype(local_domain)>,
                typename mss_components_t::cache_sequence_t,
                Grid,
                typename mss_components_t::execution_engine_t>
                run_functor_args_t;

            // now the corresponding backend has to execute all the functors of the mss
            mss_loop<run_functor_args_t>(m_backend_target, local_domain, m_grid, m_execution_info);
        }
    };

    template <class MssComponentsArray, class BackendTarget, class LocalDomains, class Grid, class ExecutionInfo>
    GT_FUNCTION_HOST mss_functor<MssComponentsArray, BackendTarget, LocalDomains, Grid, ExecutionInfo> make_mss_functor(
        BackendTarget const &backend_target,
        LocalDomains const &local_domains,
        Grid const &grid,
        ExecutionInfo const &execution_info) {
        return {backend_target, local_domains, grid, execution_info};
    }
} // namespace gridtools
