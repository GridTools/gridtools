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

#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/next_prior.hpp>

#include "../../execution_policy.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "stencil-composition/backend_mic/iterate_domain_mic.hpp"
#include "stencil-composition/iterate_domain.hpp"
#include "common/generic_metafunctions/for_each.hpp"
#include "common/generic_metafunctions/meta.hpp"
#include "execinfo_mic.hpp"

namespace gridtools {

    namespace _impl {

        /**
         * @brief Simplified copy of boost::mpl::for_each for looping over loop_intervals. Needed because ICC can not
         * vectorize the original boost::mpl::for_each.
         */
        template < bool done = true >
        struct boost_mpl_for_each_mic_impl {
            template < typename Iterator, typename LastIterator, typename F >
            GT_FUNCTION static void execute(F const &) {}
        };

        template <>
        struct boost_mpl_for_each_mic_impl< false > {
            template < typename Iterator, typename LastIterator, typename F >
            GT_FUNCTION static void execute(F const &f) {
                using arg = typename ::boost::mpl::deref< Iterator >::type;
                using next = typename ::boost::mpl::next< Iterator >::type;

                f(arg{});

                boost_mpl_for_each_mic_impl<::boost::is_same< next, LastIterator >::value >::template execute< next,
                    LastIterator >(f);
            }
        };

        template < typename Sequence, typename F >
        GT_FUNCTION void boost_mpl_for_each_mic(F const &f) {
            using first = typename ::boost::mpl::begin< Sequence >::type;
            using last = typename ::boost::mpl::end< Sequence >::type;

            boost_mpl_for_each_mic_impl<::boost::is_same< first, last >::value >::template execute< first, last >(f);
        }

        /**
         * @brief Meta function to check if all ESFs can be computed independently per column. This is possible if the
         * max extent is zero, i.e. there are no dependencies between the ESFs with offsets along i or j.
         */
        template < typename RunFunctorArguments >
        using enable_inner_k_fusion = std::integral_constant< bool,
            RunFunctorArguments::max_extent_t::iminus::value == 0 &&
                RunFunctorArguments::max_extent_t::iplus::value == 0 &&
                RunFunctorArguments::max_extent_t::jminus::value == 0 &&
                RunFunctorArguments::max_extent_t::jplus::value == 0 &&
                /* TODO: enable. This is currently disabled as the performance is not satisfying, probably due to
                   missing vector-sized k-caches and possibly alignment issues. */
                false >;

        constexpr int_t veclength_mic = 64 / sizeof(float_type);

        /**
         * @brief Class for inner (block-level) looping.
         */
        template < typename RunFunctorArguments, typename Interval, typename ExecutionInfo, typename Enable = void >
        class inner_functor_mic;

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and max extent 0.
         *
         * @tparam RunFunctorArguments Run functor arguments.
         * @tparam Interval K-axis interval where the functors should be executed.
         */
        template < typename RunFunctorArguments, typename Interval >
        class inner_functor_mic< RunFunctorArguments,
            Interval,
            execinfo_block_kserial_mic,
            typename std::enable_if< enable_inner_k_fusion< RunFunctorArguments >::value >::type > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mic(iterate_domain_t &it_domain,
                const grid_t &grid,
                const execinfo_block_kserial_mic &execution_info,
                int_t i_vecfirst,
                int_t i_veclast)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info), m_i_vecfirst(i_vecfirst),
                  m_i_veclast(i_veclast) {}

            /**
             * @brief Executes the corresponding functor in the given interval.
             *
             * @param index Index in the functor list of the ESF functor that should be executed.
             */
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

                const int_t k_first = m_grid.template value_at< typename iteration_policy_t::from >();
                const int_t k_last = m_grid.template value_at< typename iteration_policy_t::to >();

                /* Prefetching is only done for the first ESF, as we assume the following ESFs access the same data
                 * that's already in cache then. The prefetching distance is currently always 2 k-levels. */
                if (Index::value == 0)
                    m_it_domain.set_prefetch_distance(k_first <= k_last ? 2 : -2);

                run_esf_functor_t run_esf(m_it_domain);
                for (int_t k = k_first; iteration_policy_t::condition(k, k_last); iteration_policy_t::increment(k)) {
                    m_it_domain.set_k_block_index(k);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                    for (int_t i = m_i_vecfirst; i < m_i_veclast; ++i) {
                        m_it_domain.set_i_block_index(i);
                        run_esf(index);
                    }
                }

                m_it_domain.set_prefetch_distance(0);
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mic &m_execution_info;
            const int_t m_i_vecfirst, m_i_veclast;
        };

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         *
         * @tparam RunFunctorArguments Run functor arguments.
         * @tparam Interval K-axis interval where the functors should be executed.
         */
        template < typename RunFunctorArguments, typename Interval >
        class inner_functor_mic< RunFunctorArguments,
            Interval,
            execinfo_block_kserial_mic,
            typename std::enable_if< !enable_inner_k_fusion< RunFunctorArguments >::value >::type > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Executes the corresponding functor in the given interval.
             *
             * @param index Index in the functor list of the ESF functor that should be executed.
             */
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

                run_esf_functor_t run_esf(m_it_domain);
                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
                    for (int_t k = k_first; iteration_policy_t::condition(k, k_last);
                         iteration_policy_t::increment(k)) {
                        m_it_domain.set_k_block_index(k);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                        for (int_t i = i_first; i < i_last; ++i) {
                            m_it_domain.set_i_block_index(i);
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

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with parallel execution along k-axis.
         *
         * @tparam RunFunctorArguments Run functor arguments.
         * @tparam Interval K-axis interval where the functors should be executed.
         */
        template < typename RunFunctorArguments, typename Interval >
        class inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kparallel_mic, void > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Executes the corresponding functor on a single k-level inside the block.
             *
             * @param index Index in the functor list of the ESF functor that should be executed.
             */
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
                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
                m_it_domain.set_k_block_index(m_execution_info.k);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                    for (int_t i = i_first; i < i_last; ++i) {
                        m_it_domain.set_i_block_index(i);
                        run_esf(index);
                    }
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kparallel_mic &m_execution_info;
        };

        /**
         * @brief Class for per-block looping on a single interval.
         */
        template < typename RunFunctorArguments, typename ExecutionInfo, typename Enable = void >
        class interval_functor_mic;

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and max extent of 0.
         */
        template < typename RunFunctorArguments >
        class interval_functor_mic< RunFunctorArguments,
            execinfo_block_kserial_mic,
            typename std::enable_if< enable_inner_k_fusion< RunFunctorArguments >::value >::type > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION interval_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval.
             */
            template < typename Interval >
            GT_FUNCTION void operator()(const Interval &) const {
                using extent_t = typename RunFunctorArguments::max_extent_t;
                using functor_list_t = typename RunFunctorArguments::functor_list_t;
                using range_t = meta::make_indices< boost::mpl::size< functor_list_t >::value >;
                using inner_functor_t = inner_functor_mic< RunFunctorArguments, Interval, execinfo_block_kserial_mic >;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;

                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
                    for (int_t i_vecfirst = i_first; i_vecfirst < i_last; i_vecfirst += veclength_mic) {
                        const int_t i_veclast =
                            i_vecfirst + veclength_mic > i_last ? i_last : i_vecfirst + veclength_mic;
                        gridtools::for_each< range_t >(
                            inner_functor_t(m_it_domain, m_grid, m_execution_info, i_vecfirst, i_veclast));
                    }
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mic &m_execution_info;
        };

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         */
        template < typename RunFunctorArguments >
        class interval_functor_mic< RunFunctorArguments,
            execinfo_block_kserial_mic,
            typename std::enable_if< !enable_inner_k_fusion< RunFunctorArguments >::value >::type > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION interval_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval.
             */
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

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with parallel execution along k-axis.
         */
        template < typename RunFunctorArguments, typename Enable >
        class interval_functor_mic< RunFunctorArguments, execinfo_block_kparallel_mic, Enable > {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION interval_functor_mic(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mic &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {
                // enable ij-caches
                m_it_domain.enable_ij_caches();
            }

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval if k is inside the interval.
             */
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

        /**
         * @brief Class for executing all functors on a single block.
         */
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
                using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

                iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                gridtools::_impl::boost_mpl_for_each_mic< loop_intervals_t >(
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
