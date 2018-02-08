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
#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "../../backend_mic/execinfo_mic.hpp"
#include "stencil-composition/backend_mic/iterate_domain_mic.hpp"
#include "stencil-composition/iterate_domain.hpp"
#include "common/generic_metafunctions/for_each.hpp"
#include "common/generic_metafunctions/meta.hpp"

namespace gridtools {

    namespace _impl {

        template < typename RunFunctorArguments, typename Interval, typename ExecutionInfo >
        class inner_functor_mic;

        template < typename RunFunctorArguments, typename Interval >
        class inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kserial_mic > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            template < typename Index >
            GT_FUNCTION void operator()(const Index &index) const {
                using backend_traits_t = backend_traits_from_id< RunFunctorArguments::backend_ids_t::s_backend_id >;
                using grid_traits_t = grid_traits_from_id< RunFunctorArguments::backend_ids_t::s_grid_type_id >;

                using interval_from_t = typename index_to_level< typename Interval::first >::type;
                using interval_to_t = typename index_to_level< typename Interval::second >::type;
                using execution_type_t = typename RunFunctorArguments::execution_type_t;
                using iteration_policy_t = ::gridtools::_impl::iteration_policy< interval_from_t,
                    interval_to_t,
                    typename grid_traits_t::dim_k_t,
                    execution_type_t::type::iteration >;

                using run_esf_functor_t =
                    typename backend_traits_t::run_esf_functor_h_t::template apply< RunFunctorArguments,
                        Interval >::type;

                using extent_sizes_t = typename RunFunctorArguments::extent_sizes_t;
                using extent_t = typename boost::mpl::at< extent_sizes_t, Index >::type;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;
                const int_t k_first = m_grid.template value_at< typename iteration_policy_t::from >();
                const int_t k_last = m_grid.template value_at< typename iteration_policy_t::to >();

                m_it_domain.set_index(0, 0, 0, m_execution_info.i_first, m_execution_info.j_first);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.template set_block_index< 1 >(j);
// TODO: fix vectorization for ICC
#pragma ivdep
#pragma omp simd
                    for (int_t i = i_first; i < i_last; ++i) {
                        run_esf_functor_t run_esf(m_it_domain);
                        m_it_domain.template set_block_index< 0 >(i);
                        for (int_t k = k_first; iteration_policy_t::condition(k, k_last);
                             iteration_policy_t::increment(k)) {
                            m_it_domain.template set_block_index< 2 >(k);
                            run_esf(index);
                        }
                    }
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mic &m_execution_info;
        };

        template < typename RunFunctorArguments, typename Interval >
        class inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kparallel_mic > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            template < typename Index >
            GT_FUNCTION void operator()(const Index &index) const {
                using backend_traits_t = backend_traits_from_id< RunFunctorArguments::backend_ids_t::s_backend_id >;
                using run_esf_functor_t =
                    typename backend_traits_t::run_esf_functor_h_t::template apply< RunFunctorArguments,
                        Interval >::type;
                using extent_sizes_t = typename RunFunctorArguments::extent_sizes_t;
                using extent_t = typename boost::mpl::at< extent_sizes_t, Index >::type;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;

                run_esf_functor_t run_esf(m_it_domain);
                m_it_domain.set_index(0, 0, m_execution_info.k, m_execution_info.i_first, m_execution_info.j_first);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.template set_block_index< 1 >(j);
// TODO: fix vectorization for ICC
#pragma ivdep
#pragma omp simd
                    for (int_t i = i_first; i < i_last; ++i) {
                        m_it_domain.template set_block_index< 0 >(i);
                        run_esf(index);
                    }
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kparallel_mic &m_execution_info;
        };

        template < typename RunFunctorArguments, typename ExecutionInfo >
        class interval_functor_mic;

        template < typename RunFunctorArguments >
        class interval_functor_mic< RunFunctorArguments, execinfo_block_kserial_mic > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION interval_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            template < typename Interval >
            GT_FUNCTION void operator()(const Interval &) const {
                using functor_list_t = typename RunFunctorArguments::functor_list_t;
                using range_t = meta::make_indices< boost::mpl::size< functor_list_t >::value >;
                using inner_functor_t = inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kserial_mic >;

                gridtools::for_each< range_t >(inner_functor_t(m_it_domain, m_grid, m_execution_info));
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mic &m_execution_info;
        };

        template < typename RunFunctorArguments >
        class interval_functor_mic< RunFunctorArguments, execinfo_block_kparallel_mic > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION interval_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            template < typename Interval >
            GT_FUNCTION void operator()(const Interval &) const {
                using from_t = typename index_to_level< typename Interval::first >::type;
                using to_t = typename index_to_level< typename Interval::second >::type;

                const int_t k_first = this->m_grid.template value_at< from_t >();
                const int_t k_last = this->m_grid.template value_at< to_t >();

                if (k_first <= m_execution_info.k && m_execution_info.k <= k_last) {
                    using functor_list_t = typename RunFunctorArguments::functor_list_t;
                    using range_t = meta::make_indices< boost::mpl::size< functor_list_t >::value >;
                    using inner_functor_t =
                        inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kparallel_mic >;

                    gridtools::for_each< range_t >(inner_functor_t(m_it_domain, m_grid, m_execution_info));
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kparallel_mic &m_execution_info;
        };

    } // namespace _impl

    namespace strgrid {
        template < typename RunFunctorArguments >
        class execute_kernel_functor_mic {
            using grid_t = typename RunFunctorArguments::grid_t;
            using local_domain_t = typename RunFunctorArguments::local_domain_t;
            using loop_intervals_t = typename RunFunctorArguments::loop_intervals_t;
            using reduction_data_t = typename RunFunctorArguments::reduction_data_t;

          public:
            GT_FUNCTION execute_kernel_functor_mic(
                const local_domain_t &local_domain, const grid_t &grid, reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data) {}

            template < class ExecutionInfo >
            GT_FUNCTION void operator()(const ExecutionInfo &execution_info) const {
                using backend_traits_t = backend_traits_from_id< enumtype::Mic >;
                using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

                using data_ptr_cached_t = typename iterate_domain_t::data_ptr_cached_t;
                using strides_cached_t = typename iterate_domain_t::strides_cached_t;

                iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                data_ptr_cached_t data_ptr;
                strides_cached_t strides;
                it_domain.set_data_pointer_impl(&data_ptr);
                it_domain.set_strides_pointer_impl(&strides);

                it_domain.template assign_storage_pointers< backend_traits_t >();
                it_domain.template assign_stride_pointers< backend_traits_t, strides_cached_t >();

                boost::mpl::for_each< loop_intervals_t >(
                    gridtools::_impl::interval_functor_mic< RunFunctorArguments, ExecutionInfo >(
                        it_domain, m_grid, execution_info));

                m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;
        };

    } // namespace strgrid
} // namespace gridtools
