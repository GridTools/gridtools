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
#pragma once
#include <boost/utility/enable_if.hpp>

#include "../../../common/generic_metafunctions/meta.hpp"
#include "../../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../basic_token_execution.hpp"
#include "../../grid_traits_fwd.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../esf_metafunctions.hpp"
#include "../grid_traits.hpp"
#include "./iterate_domain_host.hpp"
#include "./run_esf_functor_host.hpp"

namespace gridtools {

    namespace icgrid {

        template <typename RunFunctorArguments, typename IterateDomain, typename Grid, typename Extent>
        struct color_execution_functor {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;
            typedef typename RunFunctorArguments::esf_sequence_t esf_sequence_t;

          private:
            IterateDomain &m_it_domain;
            Grid const &m_grid;
            uint_t m_loop_size;

          public:
            color_execution_functor(IterateDomain &it_domain, Grid const &grid, uint_t loop_size)
                : m_it_domain(it_domain), m_grid(grid), m_loop_size(loop_size) {}

            template <typename Index>
            void operator()(Index const &,
                typename boost::enable_if<typename esf_sequence_contains_color<esf_sequence_t,
                    color_type<Index::value>>::type>::type * = 0) const {

                for (uint_t j = 0; j != m_loop_size; ++j) {
                    auto memorized_index = m_it_domain.index();

                    // we fill the run_functor_arguments with the current color being processed
                    using run_functor_arguments_t = GT_META_CALL(meta::replace,
                        (RunFunctorArguments, typename RunFunctorArguments::color_t, color_type<(uint_t)Index::value>));

                    run_functors_on_interval<run_functor_arguments_t, run_esf_functor_host>(m_it_domain, m_grid);
                    m_it_domain.set_index(memorized_index);
                    m_it_domain.increment_j();
                }
                m_it_domain.increment_j(-m_loop_size);
                m_it_domain.increment_c();
            }
            template <typename Index>
            void operator()(Index const &,
                typename boost::disable_if<typename esf_sequence_contains_color<esf_sequence_t,
                    color_type<Index::value>>::type>::type * = 0) const {
                // If there is no ESF in the sequence matching the color, we skip execution and simply increment the
                // color iterator
                m_it_domain.increment_c();
            }
        };
        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_host {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::esf_sequence_t esf_sequence_t;

            typedef typename extract_esf_location_type<esf_sequence_t>::type location_type_t;

            using n_colors_t = typename location_type_t::n_colors;

            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            // in the host backend there should be only one esf per mss
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size<typename RunFunctorArguments::extent_sizes_t>::value == 1), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::back<typename RunFunctorArguments::extent_sizes_t>::type extent_t;
            GRIDTOOLS_STATIC_ASSERT((is_extent<extent_t>::value), GT_INTERNAL_ERROR);

            using iterate_domain_arguments_t = iterate_domain_arguments<typename RunFunctorArguments::backend_ids_t,
                local_domain_t,
                typename RunFunctorArguments::esf_sequence_t,
                typename RunFunctorArguments::extent_sizes_t,
                typename RunFunctorArguments::max_extent_t,
                typename RunFunctorArguments::cache_sequence_t,
                grid_t>;

            using iterate_domain_t = iterate_domain_host<iterate_domain_arguments_t>;

            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
            typedef typename boost::mpl::front<loop_intervals_t>::type interval;
            typedef typename index_to_level<typename interval::first>::type from;
            typedef typename index_to_level<typename interval::second>::type to;
            typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy_t;

            template <class ReductionData>
            execute_kernel_functor_host(const local_domain_t &local_domain,
                const grid_t &grid,
                ReductionData &&,
                uint_t block_size_i,
                uint_t block_size_j,
                uint_t block_no_i,
                uint_t block_no_j)
                : m_local_domain(local_domain),
                  m_grid(grid), m_size{block_size_i + extent_t::iplus::value - extent_t::iminus::value,
                                    block_size_j + extent_t::jplus::value - extent_t::jminus::value},
                  m_block_no{block_no_i, block_no_j} {}

            void operator()() const {
                iterate_domain_t it_domain(m_local_domain, m_grid.grid_topology());

                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    m_block_no,
                    {extent_t::iminus::value,
                        extent_t::jminus::value,
                        static_cast<int_t>(
                            m_grid.template value_at<typename iteration_policy_t::from>() - m_grid.k_min())});

                for (uint_t i = 0; i != m_size.i; ++i) {
                    boost::mpl::for_each<boost::mpl::range_c<uint_t, 0, n_colors_t::value>>(
                        color_execution_functor<RunFunctorArguments, iterate_domain_t, grid_t, extent_t>{
                            it_domain, m_grid, m_size.j});
                    it_domain.template increment_c<-int_t(n_colors_t::value)>();
                    it_domain.increment_i();
                }
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            pos3<uint_t> m_size;
            pos3<uint_t> m_block_no;
        };
    } // namespace icgrid
} // namespace gridtools
