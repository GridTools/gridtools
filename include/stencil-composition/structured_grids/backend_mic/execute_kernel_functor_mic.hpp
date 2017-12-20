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
 * execute_kernel_functor_mic.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "stencil-composition/backend_mic/iterate_domain_mic.hpp"
#include "stencil-composition/iterate_domain.hpp"

namespace gridtools {

    namespace _impl {

        template < typename ExecutionEngine, typename RunFunctorArguments, typename Interval >
        struct functor_loop_mic {
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;
            typedef typename RunFunctorArguments::extent_sizes_t extent_sizes_t;
            typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef
                typename backend_traits_from_id< RunFunctorArguments::backend_ids_t::s_backend_id >::run_esf_functor_h_t
                    run_esf_functor_h_t;

            GT_FUNCTION
            functor_loop_mic(iterate_domain_t &it_domain,
                grid_t const &grid,
                const int_t i_first,
                const int_t j_first,
                const int_t i_size,
                const int_t j_size)
                : m_it_domain(it_domain), m_grid(grid), m_i_first(i_first), m_j_first(j_first), m_i_size(i_size),
                  m_j_size(j_size) {}

            template < typename Index >
            GT_FUNCTION void operator()(Index const &index) const {
                typedef typename index_to_level< typename Interval::first >::type from_t;
                typedef typename index_to_level< typename Interval::second >::type to_t;

                typedef typename boost::mpl::at< extent_sizes_t, Index >::type extent_t;

                // check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index< typename grid_t::axis_type::FromLevel >::value <= Interval::first::value &&
                        level_to_index< typename grid_t::axis_type::ToLevel >::value >= Interval::second::value),
                    "the k interval exceeds the axis you specified for the grid instance");

                typedef iteration_policy< from_t,
                    to_t,
                    typename grid_traits_from_id< RunFunctorArguments::backend_ids_t::s_grid_type_id >::dim_k_t,
                    ExecutionEngine::type::iteration >
                    iteration_policy_t;

                constexpr int_t ifirst = extent_t::iminus::value;
                const int_t ilast = m_i_size + extent_t::iplus::value;
                constexpr int_t jfirst = extent_t::jminus::value;
                const int_t jlast = m_j_size + extent_t::jplus::value;
                const int_t kfirst = m_grid.template value_at< typename iteration_policy_t::from >();
                const int_t klast = m_grid.template value_at< typename iteration_policy_t::to >();

                typedef typename run_esf_functor_h_t::template apply< RunFunctorArguments, Interval >::type
                    run_esf_functor_t;

                run_esf_functor_t run_esf(m_it_domain);
                for (int_t k = kfirst; iteration_policy_t::condition(k, klast); iteration_policy_t::increment(k)) {
                    for (int_t j = jfirst; j < jlast; ++j) {
                        m_it_domain.set_index(0, j, k, m_i_first, m_j_first);

#pragma ivdep
#pragma omp simd
                        for (int_t i = ifirst; i < ilast; ++i) {
                            m_it_domain.template set_block_index< 0 >(i);
                            run_esf(index);

#if defined(__INTEL_COMPILER) && !defined(GT_NO_CONSTEXPR_ACCESSES)
#warning \
    "The usage of the constexpr constructors of accessor_base, tuple_offset and dimension together with the Intel compiler can lead to incorrect code generation in this loop."
#endif
                        }
                    }
                }
            }

          protected:
            iterate_domain_t &m_it_domain;
            grid_t const &m_grid;
            int_t m_i_first, m_j_first, m_i_size, m_j_size;
        };

        template < typename ExecutionEngine, typename RunFunctorArguments >
        struct block_loop_mic {
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;
            typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;

            GT_FUNCTION
            block_loop_mic(iterate_domain_t &it_domain,
                grid_t const &grid,
                int_t i_first,
                int_t j_first,
                int_t i_size,
                int_t j_size)
                : m_it_domain(it_domain), m_grid(grid), m_i_first(i_first), m_j_first(j_first), m_i_size(i_size),
                  m_j_size(j_size) {}

            template < typename Interval >
            GT_FUNCTION void operator()(Interval const &) const {
                typedef functor_loop_mic< ExecutionEngine, RunFunctorArguments, Interval > functor_loop_mic_t;

                boost::mpl::for_each< boost::mpl::range_c< int, 0, boost::mpl::size< functor_list_t >::value > >(
                    functor_loop_mic_t(m_it_domain, m_grid, m_i_first, m_j_first, m_i_size, m_j_size));
            }

          protected:
            iterate_domain_t &m_it_domain;
            grid_t const &m_grid;
            int_t m_i_first, m_j_first, m_i_size, m_j_size;
        };

    } // namespace _impl

    namespace strgrid {

        /**
        * @brief main functor that setups the CUDA kernel for a MSS and launchs it
        * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
        */
        template < typename RunFunctorArguments >
        struct execute_kernel_functor_mic {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;
            typedef typename reduction_data_t::reduction_type_t reduction_type_t;

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;

          public:
            /**
            * @brief core of the kernel execution
            * @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
            */
            explicit execute_kernel_functor_mic(const local_domain_t &local_domain,
                const grid_t &grid,
                reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data) {}

            void operator()() {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;
                typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
                typedef backend_traits_from_id< enumtype::Mic > backend_traits_t;

#pragma omp parallel
                {
                    typename iterate_domain_t::data_ptr_cached_t data_pointer;
                    typedef typename iterate_domain_t::strides_cached_t strides_t;
                    strides_t strides;

                    iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                    it_domain.set_data_pointer_impl(&data_pointer);
                    it_domain.set_strides_pointer_impl(&strides);

                    it_domain.template assign_storage_pointers< backend_traits_t >();
                    it_domain.template assign_stride_pointers< backend_traits_t, strides_t >();

                    constexpr int_t i_block_size = GT_DEFAULT_TILE_I;
                    constexpr int_t j_block_size = GT_DEFAULT_TILE_J;

                    const int_t i_grid_size = m_grid.i_high_bound() - m_grid.i_low_bound() + 1;
                    const int_t j_grid_size = m_grid.j_high_bound() - m_grid.j_low_bound() + 1;

                    const int i_blocks = (i_grid_size + i_block_size - 1) / i_block_size;
                    const int j_blocks = (j_grid_size + j_block_size - 1) / j_block_size;

#pragma omp for collapse(2)
                    for (int_t bj = 0; bj < j_blocks; ++bj) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            const int_t i_first = bi * i_block_size + m_grid.i_low_bound();
                            const int_t j_first = bj * j_block_size + m_grid.j_low_bound();

                            const int_t i_bs = (bi == i_blocks - 1) ? i_grid_size - bi * i_block_size : i_block_size;
                            const int_t j_bs = (bj == j_blocks - 1) ? j_grid_size - bj * j_block_size : j_block_size;

                            boost::mpl::for_each< loop_intervals_t >(
                                ::gridtools::_impl::block_loop_mic< execution_type_t, RunFunctorArguments >(
                                    it_domain, m_grid, i_first, j_first, i_bs, j_bs));
                        }
                    }

                    m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                }
                m_reduction_data.reduce();
            }
        };
    } // namespace strgrid
} // namespace gridtools
