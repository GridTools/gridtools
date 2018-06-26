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
#include "../../backend_host/iterate_domain_host.hpp"
#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"

namespace gridtools {

    namespace strgrid {

        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_host {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;
            typedef typename reduction_data_t::reduction_type_t reduction_type_t;

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;
            const gridtools::array<const uint_t, 2> m_first_pos;
            const gridtools::array<const uint_t, 2> m_last_pos;
            const gridtools::array<const uint_t, 2> m_block_id;

          public:
            /**
             * @brief core of the kernel execution
             */
            explicit execute_kernel_functor_host(const local_domain_t &local_domain,
                const grid_t &grid,
                reduction_data_t &reduction_data,
                const uint_t first_i,
                const uint_t first_j,
                const uint_t last_i,
                const uint_t last_j,
                const uint_t block_idx_i,
                const uint_t block_idx_j)
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data),
                  m_first_pos{first_i, first_j}, m_last_pos{last_i, last_j}, m_block_id{block_idx_i, block_idx_j} {}

            // Naive strategy
            explicit execute_kernel_functor_host(
                const local_domain_t &local_domain, const grid_t &grid, reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid),
                  m_reduction_data(reduction_data), m_first_pos{grid.i_low_bound(), grid.j_low_bound()},
                  m_last_pos{grid.i_high_bound() - grid.i_low_bound(), grid.j_high_bound() - grid.j_low_bound()},
                  m_block_id{0, 0} {}

            void operator()() {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;

                // in the host backend there should be only one esf per mss
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size<typename RunFunctorArguments::extent_sizes_t>::value == 1), GT_INTERNAL_ERROR);
                typedef typename boost::mpl::back<typename RunFunctorArguments::extent_sizes_t>::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((is_extent<extent_t>::value), GT_INTERNAL_ERROR);

                typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
                typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifdef VERBOSE
#pragma omp critical
                {
                    std::cout << "I loop " << m_first_pos[0] << "+" << extent_t::iminus::value << " -> "
                              << m_first_pos[0] << "+" << m_last_pos[0] << "+" << extent_t::iplus::value << "\n";
                    std::cout << "J loop " << m_first_pos[1] << "+" << extent_t::jminus::value << " -> "
                              << m_first_pos[1] << "+" << m_last_pos[1] << "+" << extent_t::jplus::value << "\n";
                    std::cout << "iminus::value: " << extent_t::iminus::value << std::endl;
                    std::cout << "iplus::value: " << extent_t::iplus::value << std::endl;
                    std::cout << "jminus::value: " << extent_t::jminus::value << std::endl;
                    std::cout << "jplus::value: " << extent_t::jplus::value << std::endl;
                    std::cout << "block_id_i: " << m_block_id[0] << std::endl;
                    std::cout << "block_id_j: " << m_block_id[1] << std::endl;
                }
#endif

                typename iterate_domain_t::data_ptr_cached_t data_pointer;
                typedef typename iterate_domain_t::strides_cached_t strides_t;
                strides_t strides;

                iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                it_domain.set_data_pointer_impl(&data_pointer);
                it_domain.set_strides_pointer_impl(&strides);

                it_domain.template assign_storage_pointers<backend_traits_t>();
                it_domain.template assign_stride_pointers<backend_traits_t, strides_t>();

                typedef typename boost::mpl::front<loop_intervals_t>::type interval;
                typedef typename index_to_level<typename interval::first>::type from;
                typedef typename index_to_level<typename interval::second>::type to;
                typedef ::gridtools::_impl::iteration_policy<from, to, execution_type_t::type::iteration>
                    iteration_policy_t;

                const int_t ifirst = m_first_pos[0] + extent_t::iminus::value;
                const int_t ilast = m_first_pos[0] + m_last_pos[0] + extent_t::iplus::value;
                const int_t jfirst = m_first_pos[1] + extent_t::jminus::value;
                const int_t jlast = m_first_pos[1] + m_last_pos[1] + extent_t::jplus::value;

                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    {m_block_id[0], m_block_id[1], 0},
                    {extent_t::iminus::value,
                        extent_t::jminus::value,
                        m_grid.template value_at<typename iteration_policy_t::from>() - m_grid.k_min()});

                // run the nested ij loop
                typename iterate_domain_t::array_index_t irestore_index, jrestore_index;
                for (int_t i = ifirst; i <= ilast; ++i) {
#if defined(VERBOSE) && !defined(NDEBUG)
                    std::cout << "iteration " << i << ", index i" << std::endl;
#endif
                    irestore_index = it_domain.index();
                    for (int_t j = jfirst; j <= jlast; ++j) {
#if defined(VERBOSE) && !defined(NDEBUG)
                        std::cout << "iteration " << j << ", index j" << std::endl;
#endif
                        jrestore_index = it_domain.index();
                        boost::mpl::for_each<loop_intervals_t>(
                            ::gridtools::_impl::run_f_on_interval<execution_type_t, RunFunctorArguments>{
                                it_domain, m_grid});
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
