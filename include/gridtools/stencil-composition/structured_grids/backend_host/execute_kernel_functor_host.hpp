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
 * execute_kernel_functor_host.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "../../backend_host/basic_token_execution_host.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../positional_iterate_domain.hpp"
#include "./iterate_domain_host.hpp"
#include "./run_esf_functor_host.hpp"

namespace gridtools {

    namespace strgrid {

        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_host {
          private:
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;
            typedef typename reduction_data_t::reduction_type_t reduction_type_t;

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
                grid_t,
                typename RunFunctorArguments::is_reduction_t,
                reduction_type_t>;
            using iterate_domain_host_t = iterate_domain_host<iterate_domain_arguments_t>;
            using iterate_domain_t = typename conditional_t<local_domain_is_stateful<local_domain_t>::value,
                meta::lazy::id<positional_iterate_domain<iterate_domain_host_t>>,
                meta::lazy::id<iterate_domain_host_t>>::type;

            typedef backend_traits_from_id<platform::x86> backend_traits_t;

            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            typedef typename boost::mpl::front<loop_intervals_t>::type interval;
            typedef typename index_to_level<typename interval::first>::type from;
            typedef typename index_to_level<typename interval::second>::type to;
            typedef ::gridtools::_impl::iteration_policy<from, to, execution_type_t::iteration> iteration_policy_t;

            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;
            pos3<uint_t> m_size;
            pos3<uint_t> m_block_no;

          public:
            execute_kernel_functor_host(const local_domain_t &local_domain,
                const grid_t &grid,
                reduction_data_t &reduction_data,
                uint_t block_size_i,
                uint_t block_size_j,
                uint_t block_no_i,
                uint_t block_no_j)
                : m_local_domain(local_domain), m_grid(grid),
                  m_reduction_data(reduction_data), m_size{block_size_i + extent_t::iplus::value -
                                                               extent_t::iminus::value,
                                                        block_size_j + extent_t::jplus::value -
                                                            extent_t::jminus::value},
                  m_block_no{block_no_i, block_no_j} {}

            void operator()() const {

                iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    m_block_no,
                    {extent_t::iminus::value,
                        extent_t::jminus::value,
                        static_cast<int_t>(
                            m_grid.template value_at<typename iteration_policy_t::from>() - m_grid.k_min())});

                // run the nested ij loop
                typename iterate_domain_t::array_index_t irestore_index, jrestore_index;
                for (uint_t i = 0; i != m_size.i; ++i) {
                    irestore_index = it_domain.index();
                    for (uint_t j = 0; j != m_size.j; ++j) {
                        jrestore_index = it_domain.index();
                        run_functors_on_interval<RunFunctorArguments, run_esf_functor_host>(it_domain, m_grid);
                        it_domain.set_index(jrestore_index);
                        it_domain.increment_j();
                    }
                    it_domain.set_index(irestore_index);
                    it_domain.increment_i();
                }
                m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                m_reduction_data.reduce();
            }
        };
    } // namespace strgrid
} // namespace gridtools
