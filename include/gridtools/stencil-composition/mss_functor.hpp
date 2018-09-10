/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
#include "../common/generic_metafunctions/type_traits.hpp"
#include "./backend_ids.hpp"
#include "./backend_traits_fwd.hpp"
#include "./grid.hpp"
#include "./local_domain.hpp"
#include "./mss_components.hpp"
#include "./mss_components_metafunctions.hpp"
#include "./reductions/reduction_data.hpp"
#include "./run_functor_arguments.hpp"

namespace gridtools {
    /**
     * @brief functor that executes all the functors contained within the mss
     */
    template <typename MssComponentsArray,
        typename Grid,
        typename LocalDomains,
        typename BackendIds,
        typename ReductionData,
        typename ExecutionInfo>
    struct mss_functor {
      private:
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<LocalDomains, is_local_domain>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssComponentsArray, is_mss_components>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

        LocalDomains const &m_local_domains;
        const Grid &m_grid;
        ReductionData &m_reduction_data;
        const ExecutionInfo m_execution_info;

      public:
        mss_functor(LocalDomains const &local_domains,
            const Grid &grid,
            ReductionData &reduction_data,
            const ExecutionInfo &execution_info)
            : m_local_domains(local_domains), m_grid(grid), m_reduction_data(reduction_data),
              m_execution_info(execution_info) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the
         * operations defined on that functor.
         */
        template <typename Index>
        GT_FUNCTION_HOST void operator()(Index) const {
            GRIDTOOLS_STATIC_ASSERT((Index::value < boost::mpl::size<MssComponentsArray>::value), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::at<MssComponentsArray, Index>::type mss_components_t;

            auto const &local_domain = boost::fusion::at<Index>(m_local_domains);

            // wrapping all the template arguments in a single container

            using backend_traits_t = backend_traits_from_id<typename BackendIds::backend_id_t>;

            // perform some checks concerning the reduction types
            typedef run_functor_arguments<BackendIds,
                typename mss_components_t::linear_esf_t,
                typename mss_components_t::loop_intervals_t,
                typename mss_components_t::extent_sizes_t,
                decay_t<decltype(local_domain)>,
                typename mss_components_t::cache_sequence_t,
                Grid,
                typename mss_components_t::execution_engine_t,
                typename mss_components_is_reduction<mss_components_t>::type,
                ReductionData>
                run_functor_args_t;

            // now the corresponding backend has to execute all the functors of the mss
            backend_traits_t::template mss_loop<run_functor_args_t>::template run(
                local_domain, m_grid, m_reduction_data, m_execution_info);
        }
    };
} // namespace gridtools
