/*
 * execute_kernel_functor_host.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "stencil-composition/backend_host/iterate_domain_host.hpp"
#include "stencil-composition/loop_hierarchy.hpp"
#include "../../iteration_policy.hpp"
#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"

namespace gridtools {

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

            GRIDTOOLS_STATIC_ASSERT(
                (is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");
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
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data)
#ifdef CXX11_ENABLED
                  ,
                  m_first_pos{first_i, first_j}, m_last_pos{last_i, last_j}, m_block_id {
                block_idx_i, block_idx_j
            }
#else
                  ,
                  m_first_pos(first_i, first_j), m_last_pos(last_i, last_j), m_block_id(block_idx_i, block_idx_j)
#endif
            {}

            // Naive strategy
            explicit execute_kernel_functor_host(
                const local_domain_t &local_domain, const grid_t &grid, reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data)
#ifdef CXX11_ENABLED
                  ,
                  m_first_pos{grid.i_low_bound(), grid.j_low_bound()},
                  m_last_pos{grid.i_high_bound() - grid.i_low_bound(), grid.j_high_bound() - grid.j_low_bound()},
                  m_block_id {
                0, 0
            }
#else
                  ,
                  m_first_pos(grid.i_low_bound(), grid.j_low_bound()),
                  m_last_pos(grid.i_high_bound() - grid.i_low_bound(), grid.j_high_bound() - grid.j_low_bound()),
                  m_block_id(0, 0)
#endif
            {}

            void operator()() {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;

                // in the host backend there should be only one esf per mss
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename RunFunctorArguments::extent_sizes_t >::value == 1),
                    "Internal Error: wrong size");
                typedef typename boost::mpl::back< typename RunFunctorArguments::extent_sizes_t >::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((is_extent< extent_t >::value), "Internal Error: wrong type");

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

                typename iterate_domain_t::data_pointer_array_t data_pointer;
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
                typedef _impl::iteration_policy< from,
                    to,
                    typename ::gridtools::grid_traits_from_id< enumtype::structured >::dim_k_t,
                    execution_type_t::type::iteration > iteration_policy_t;

                typedef array< int_t, iterate_domain_t::N_META_STORAGES > array_t;
                loop_hierarchy< array_t, loop_item< 0, int_t, 1 >, loop_item< 1, int_t, 1 > > ij_loop(
                    (int_t)(m_first_pos[0] + extent_t::iminus::value),
                    (int_t)(m_first_pos[0] + m_last_pos[0] + extent_t::iplus::value),
                    (int_t)(m_first_pos[1] + extent_t::jminus::value),
                    (int_t)(m_first_pos[1] + m_last_pos[1] + extent_t::jplus::value));

                // reset the index
                it_domain.set_index(0);
                ij_loop.initialize(it_domain, m_block_id);

                // define the kernel functor
                typedef innermost_functor< loop_intervals_t,
                    _impl::run_f_on_interval< execution_type_t, RunFunctorArguments >,
                    iterate_domain_t,
                    grid_t,
                    iteration_policy_t > innermost_functor_t;

                // instantiate the kernel functor
                innermost_functor_t f(it_domain, m_grid);

                // run the nested ij loop
                ij_loop.apply(it_domain, f);
                m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                m_reduction_data.reduce();
            }
        };
    } // namespace strgrid
} // namespace gridtools
