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
#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "stencil-composition/backend_host/iterate_domain_host.hpp"

namespace gridtools {

    namespace _impl {
        template < ushort_t Index, typename IterateDomain, typename VT >
        typename boost::enable_if< typename is_positional_iterate_domain< IterateDomain >::type, void >::type
            GT_FUNCTION
            reset_index_if_positional(IterateDomain &itdom, VT value) {
            itdom.template reset_positional_index< Index >(value);
        }
        template < ushort_t Index, typename IterateDomain, typename VT >
        typename boost::disable_if< typename is_positional_iterate_domain< IterateDomain >::type, void >::type
            GT_FUNCTION
            reset_index_if_positional(IterateDomain &, VT) {}
    } // namespace _impl

    namespace strgrid {

        /**
        * @brief main functor that setups the CUDA kernel for a MSS and launchs it
        * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
        */
        template < typename RunFunctorArguments >
        struct execute_kernel_functor_host {
            /**
            * @brief functor implementing the kernel executed in the innermost loop
            * This functor contains the portion of the code executed in the innermost loop. In this case it
            * is the loop over the third dimension (k), but the generality of the loop hierarchy implementation
            * allows to easily generalize this.
            */
            template < typename LoopIntervals,
                typename RunOnInterval,
                typename IterateDomain,
                typename Grid,
                typename IterationPolicy >
            struct innermost_functor {

              private:
                IterateDomain &m_it_domain;
                const Grid &m_grid;

              public:
                IterateDomain const &it_domain() const { return m_it_domain; }

                innermost_functor(IterateDomain &it_domain, const Grid &grid) : m_it_domain(it_domain), m_grid(grid) {}

                void operator()() const {
                    m_it_domain.template initialize< 2 >(m_grid.template value_at< typename IterationPolicy::from >());

                    boost::mpl::for_each< LoopIntervals >(RunOnInterval(m_it_domain, m_grid));
                }
            };

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;
            typedef typename reduction_data_t::reduction_type_t reduction_type_t;

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;
            const gridtools::array< const uint_t, 2 > m_first_pos;
            const gridtools::array< const uint_t, 2 > m_last_pos;
            const gridtools::array< const uint_t, 2 > m_block_id;

          public:
            /**
            * @brief core of the kernel execution
            * @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
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
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data),
                  m_first_pos{grid.i_low_bound(), grid.j_low_bound()},
                  m_last_pos{grid.i_high_bound() - grid.i_low_bound(), grid.j_high_bound() - grid.j_low_bound()},
                  m_block_id{0, 0} {}

            void operator()() {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;

                // in the host backend there should be only one esf per mss
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size< typename RunFunctorArguments::extent_sizes_t >::value == 1), GT_INTERNAL_ERROR);
                typedef typename boost::mpl::back< typename RunFunctorArguments::extent_sizes_t >::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((is_extent< extent_t >::value), GT_INTERNAL_ERROR);

                typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
                typedef backend_traits_from_id< enumtype::Host > backend_traits_t;
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

                it_domain.template assign_storage_pointers< backend_traits_t >();
                it_domain.template assign_stride_pointers< backend_traits_t, strides_t >();

                typedef typename boost::mpl::front< loop_intervals_t >::type interval;
                typedef typename index_to_level< typename interval::first >::type from;
                typedef typename index_to_level< typename interval::second >::type to;
                typedef ::gridtools::_impl::iteration_policy< from,
                    to,
                    typename ::gridtools::grid_traits_from_id< enumtype::structured >::dim_k_t,
                    execution_type_t::type::iteration > iteration_policy_t;

                const int_t ifirst = m_first_pos[0] + extent_t::iminus::value;
                const int_t ilast = m_first_pos[0] + m_last_pos[0] + extent_t::iplus::value;
                const int_t jfirst = m_first_pos[1] + extent_t::jminus::value;
                const int_t jlast = m_first_pos[1] + m_last_pos[1] + extent_t::jplus::value;

                // reset the index
                it_domain.reset_index();
                it_domain.template initialize< 0 >(ifirst, m_block_id[0]);
                it_domain.template initialize< 1 >(jfirst, m_block_id[1]);

                // define the kernel functor
                typedef innermost_functor< loop_intervals_t,
                    ::gridtools::_impl::run_f_on_interval< execution_type_t, RunFunctorArguments >,
                    iterate_domain_t,
                    grid_t,
                    iteration_policy_t > innermost_functor_t;

                // instantiate the kernel functor
                innermost_functor_t f(it_domain, m_grid);

                // run the nested ij loop
                typename iterate_domain_t::array_index_t irestore_index, jrestore_index;
                for (int_t i = ifirst; i <= ilast; ++i) {
#if defined(VERBOSE) && !defined(NDEBUG)
                    std::cout << "iteration " << i << ", index i" << std::endl;
#endif
                    ::gridtools::_impl::reset_index_if_positional< 0 >(it_domain, i);
                    irestore_index = it_domain.index();
                    for (int_t j = jfirst; j <= jlast; ++j) {
#if defined(VERBOSE) && !defined(NDEBUG)
                        std::cout << "iteration " << j << ", index j" << std::endl;
#endif
                        ::gridtools::_impl::reset_index_if_positional< 1 >(it_domain, j);
                        jrestore_index = it_domain.index();
                        f();
                        it_domain.set_index(jrestore_index);
                        it_domain.template increment< 1, static_uint< 1 > >();
                    }
                    it_domain.set_index(irestore_index);
                    it_domain.template increment< 0, static_uint< 1 > >();
                }
                m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                m_reduction_data.reduce();
            }
        };
    } // namespace strgrid
} // namespace gridtools
