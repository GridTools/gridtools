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

/**
@file
Implementation of the k loop execution policy
The policies which are currently considered are
 - forward: the k loop is executed upward, increasing the value of the iterator on k. This is the option to be used when
the stencil operations at level k depend on the fields at level k-1 (forward substitution).
 - backward: the k loop is executed downward, decreasing the value of the iterator on k. This is the option to be used
when the stencil operations at level k depend on the fields at level k+1 (backward substitution).
 - parallel: the operations on each k level are executed in parallel. This is feasable only if there are no dependencies
between levels.
*/

#pragma once

#include <boost/mpl/for_each.hpp>
#include <boost/mpl/has_key.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/generic_metafunctions/meta.hpp"

#include "grid_traits_fwd.hpp"
#include "interval.hpp"
#include "iteration_policy.hpp"
#include "level.hpp"

namespace gridtools {
    namespace _impl {

        template <class ItDomain, class RunFunctorArguments, class Interval, class Impl>
        struct run_esf_functor_helper {
            GRIDTOOLS_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArguments>::value, GT_INTERNAL_ERROR);

            template <class Index>
            using has_interval =
                boost::mpl::has_key<typename esf_arguments<RunFunctorArguments, Index>::interval_map_t, Interval>;

            ItDomain &m_domain;

            template <class Index>
            GT_FUNCTION enable_if_t<has_interval<Index>::value> operator()(Index) const {
                using esf_arguments_t = esf_arguments<RunFunctorArguments, Index>;
                using interval_map_t = typename esf_arguments_t::interval_map_t;
                using interval_type = typename boost::mpl::at<interval_map_t, Interval>::type;
                Impl{}.template operator()<interval_type, esf_arguments_t>(m_domain);
            }

            template <class Index>
            GT_FUNCTION enable_if_t<!has_interval<Index>::value> operator()(Index) const {}
        };

        /**
           @brief basic token of execution responsible of handling the discretization over the vertical dimension. This
           may be done with a loop over k or by partitoning the k axis and executing in parallel, depending on the
           execution_policy defined in the multi-stage stencil. The base class is then specialized using the CRTP
           pattern for the different policies.
        */
        /**
           @brief   Execution kernel containing the loop over k levels
        */
        template <typename RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
        struct run_f_on_interval {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArguments::execution_type_t::type execution_engine;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;

            ItDomain &m_domain;
            Grid const &m_grid;

            // TODO(anstaf): refactor k-caches related code to make it more self descriptive. All caches in the context
            // of this function are k-caches.
            template <typename IterationPolicy, typename Interval>
            GT_FUNCTION void k_loop(int_t from, int_t to) const {
                assert(to >= from);

                const bool in_domain =
                    m_domain.template is_thread_in_domain<typename RunFunctorArguments::max_extent_t>();

                run_esf_functor_helper<ItDomain, RunFunctorArguments, Interval, RunEsfFunctor> fun{m_domain};

                if (in_domain)
                    m_domain.template begin_fill<IterationPolicy>();

                for (int_t k = from; k <= to; ++k, IterationPolicy::increment(m_domain)) {
                    if (in_domain) {
                        const int_t lev = (IterationPolicy::value == enumtype::backward) ? (to - k) + from : k;
                        m_domain.template fill_caches<IterationPolicy>(lev, m_grid);
                    }

                    gridtools::for_each<GT_META_CALL(meta::make_indices_c, boost::mpl::size<functor_list_t>::value)>(
                        fun);

                    if (in_domain) {

                        const int_t lev = (IterationPolicy::value == enumtype::backward) ? (to - k) + from : k;

                        m_domain.template flush_caches<IterationPolicy>(lev, m_grid);
                        m_domain.template slide_caches<IterationPolicy>();
                    }
                }
                if (in_domain)
                    m_domain.template final_flush<IterationPolicy>();
            }

            template <typename Interval>
            GT_FUNCTION void operator()(Interval const &) const {
                typedef typename index_to_level<typename Interval::first>::type from_t;
                typedef typename index_to_level<typename Interval::second>::type to_t;

                // check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index<typename Grid::axis_type::FromLevel>::value <= Interval::first::value &&
                        level_to_index<typename Grid::axis_type::ToLevel>::value >= Interval::second::value),
                    "the k interval exceeds the axis you specified for the grid instance");

                typedef iteration_policy<from_t, to_t, execution_engine::iteration> iteration_policy_t;

                uint_t const from = m_grid.template value_at<from_t>();
                uint_t const to = m_grid.template value_at<to_t>();

                k_loop<iteration_policy_t, Interval>(from, to);
            }
        };
    } // namespace _impl

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FUNCTION void run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        boost::mpl::for_each<typename RunFunctorArguments::loop_intervals_t>(
            _impl::run_f_on_interval<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{it_domain, grid});
    }

} // namespace gridtools
