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
#include "stencil-composition/backend_mic/iterate_domain_mic.hpp"
#include "stencil-composition/iterate_domain.hpp"

namespace gridtools {

    namespace _impl {
        static int prefetch_distance = 0;

        template < typename ExecutionEngine, typename RunFunctorArguments >
        class run_on_block_mic {
          public:
            using grid_t = typename RunFunctorArguments::grid_t;
            using functor_list_t = typename RunFunctorArguments::functor_list_t;
            using extent_sizes_t = typename RunFunctorArguments::extent_sizes_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;
            using local_domain_t = typename RunFunctorArguments::local_domain_t;
            using run_esf_functor_h_t = typename backend_traits_from_id<
                RunFunctorArguments::backend_ids_t::s_backend_id >::run_esf_functor_h_t;

            GT_FUNCTION
            run_on_block_mic(iterate_domain_t &it_domain,
                grid_t const &grid,
                int_t i_first,
                int_t j_first,
                int_t i_size,
                int_t j_size)
                : m_it_domain(it_domain), m_grid(grid), m_i_first(i_first), m_j_first(j_first), m_i_size(i_size),
                  m_j_size(j_size) {}

          protected:
            iterate_domain_t &m_it_domain;
            grid_t const &m_grid;
            const int_t m_i_first, m_j_first, m_i_size, m_j_size;
        };

        template < typename ExecutionEngine, typename RunFunctorArguments, typename Interval >
        class run_esf_mic : public run_on_block_mic< ExecutionEngine, RunFunctorArguments > {
            using base = run_on_block_mic< ExecutionEngine, RunFunctorArguments >;

            using grid_t = typename base::grid_t;
            using functor_list_t = typename base::functor_list_t;
            using extent_sizes_t = typename base::extent_sizes_t;
            using iterate_domain_t = typename base::iterate_domain_t;
            using local_domain_t = typename base::local_domain_t;
            using run_esf_functor_h_t = typename base::run_esf_functor_h_t;

            using from_t = typename index_to_level< typename Interval::first >::type;
            using to_t = typename index_to_level< typename Interval::second >::type;
            using iteration_policy_t = iteration_policy< from_t,
                to_t,
                typename grid_traits_from_id< RunFunctorArguments::backend_ids_t::s_grid_type_id >::dim_k_t,
                ExecutionEngine::type::iteration >;
            using run_esf_functor_t =
                typename run_esf_functor_h_t::template apply< RunFunctorArguments, Interval >::type;

          public:
            using base::run_on_block_mic;

            template < typename Index >
            GT_FUNCTION void operator()(Index const &index) const {
                using extent_t = typename boost::mpl::at< extent_sizes_t, Index >::type;

                // check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index< typename grid_t::axis_type::FromLevel >::value <= Interval::first::value &&
                        level_to_index< typename grid_t::axis_type::ToLevel >::value >= Interval::second::value),
                    "the k interval exceeds the axis you specified for the grid instance");

                constexpr int_t i_first = extent_t::iminus::value;
                const int_t i_last = this->m_i_size + extent_t::iplus::value;
                constexpr int_t j_first = extent_t::jminus::value;
                const int_t j_last = this->m_j_size + extent_t::jplus::value;
                const int_t k_first = this->m_grid.template value_at< typename iteration_policy_t::from >();
                const int_t k_last = this->m_grid.template value_at< typename iteration_policy_t::to >();

                if (k_first <= k_last)
                    this->m_it_domain.set_prefetch_distance(prefetch_distance);

                run_esf_functor_t run_esf(this->m_it_domain);
                /*for (int_t k = k_first; iteration_policy_t::condition(k, k_last); iteration_policy_t::increment(k)) {
                    for (int_t j = j_first; j < j_last; ++j) {
                        this->m_it_domain.set_index(0, j, k, this->m_i_first, this->m_j_first);

#pragma ivdep
#pragma omp simd
                        for (int_t i = i_first; i < i_last; ++i) {
                            this->m_it_domain.template set_block_index< 0 >(i);
                            run_esf(index);

#if defined(__INTEL_COMPILER) && !defined(GT_NO_CONSTEXPR_ACCESSES)
#warning \
    "The usage of the constexpr constructors of accessor_base, tuple_offset and dimension together with the Intel
compiler can lead to incorrect code generation in this loop."
#endif
                        }
                    }
                }*/
                for (int_t j = j_first; j < j_last; ++j) {
                    for (int_t k = k_first; iteration_policy_t::condition(k, k_last);
                         iteration_policy_t::increment(k)) {
                        this->m_it_domain.set_index(0, j, k, this->m_i_first, this->m_j_first);
#pragma ivdep
#pragma omp simd
                        for (int_t i = i_first; i < i_last; ++i) {
                            this->m_it_domain.template set_block_index< 0 >(i);
                            run_esf(index);

#if defined(__INTEL_COMPILER) && !defined(GT_NO_CONSTEXPR_ACCESSES)
#warning \
    "The usage of the constexpr constructors of accessor_base, tuple_offset and dimension together with the Intel compiler can lead to incorrect code generation in this loop."
#endif
                        }
                    }
                }
                /*for (int_t j = j_first; j < j_last; ++j) {
                    this->m_it_domain.set_index(0, j, 0, this->m_i_first, this->m_j_first);
#pragma ivdep
#pragma omp simd
                    for (int_t i = i_first; i < i_last; ++i) {
                        auto it_domain = this->m_it_domain;
                        run_esf_functor_t run_esf(it_domain);
                        it_domain.template set_block_index< 0 >(i);
                        for (int_t k = k_first; iteration_policy_t::condition(k, k_last);
iteration_policy_t::increment(k)) {
                            it_domain.template set_block_index< 2 >(k);
                            run_esf(index);

#if defined(__INTEL_COMPILER) && !defined(GT_NO_CONSTEXPR_ACCESSES)
#warning \
    "The usage of the constexpr constructors of accessor_base, tuple_offset and dimension together with the Intel
compiler can lead to incorrect code generation in this loop."
#endif
                        }
                    }
                }*/
            }
        };

        template < typename ExecutionEngine, typename RunFunctorArguments, typename Interval >
        class run_esf_kparallel_mic : public run_on_block_mic< ExecutionEngine, RunFunctorArguments > {
          protected:
            using base = run_on_block_mic< ExecutionEngine, RunFunctorArguments >;

            using grid_t = typename base::grid_t;
            using functor_list_t = typename base::functor_list_t;
            using extent_sizes_t = typename base::extent_sizes_t;
            using iterate_domain_t = typename base::iterate_domain_t;
            using local_domain_t = typename base::local_domain_t;
            using run_esf_functor_h_t = typename base::run_esf_functor_h_t;

          public:
            GT_FUNCTION
            run_esf_kparallel_mic(iterate_domain_t &it_domain,
                grid_t const &grid,
                int_t i_first,
                int_t j_first,
                int_t i_size,
                int_t j_size,
                int_t k)
                : base(it_domain, grid, i_first, j_first, i_size, j_size), m_k(k) {}

            template < typename Index >
            GT_FUNCTION void operator()(Index const &index) const {
                using extent_t = typename boost::mpl::at< extent_sizes_t, Index >::type;

                constexpr int_t i_first = extent_t::iminus::value;
                const int_t i_last = this->m_i_size + extent_t::iplus::value;
                constexpr int_t j_first = extent_t::jminus::value;
                const int_t j_last = this->m_j_size + extent_t::jplus::value;

                typedef typename run_esf_functor_h_t::template apply< RunFunctorArguments, Interval >::type
                    run_esf_functor_t;

                run_esf_functor_t run_esf(this->m_it_domain);
                for (int_t j = j_first; j < j_last; ++j) {
                    this->m_it_domain.set_index(0, j, m_k, this->m_i_first, this->m_j_first);

#pragma ivdep
#pragma omp simd
                    for (int_t i = i_first; i < i_last; ++i) {
                        this->m_it_domain.template set_block_index< 0 >(i);
                        run_esf(index);

#if defined(__INTEL_COMPILER) && !defined(GT_NO_CONSTEXPR_ACCESSES)
#warning \
    "The usage of the constexpr constructors of accessor_base, tuple_offset and dimension together with the Intel compiler can lead to incorrect code generation in this loop."
#endif
                    }
                }
            }

          protected:
            int_t m_k;
        };

        template < typename F >
        GT_FUNCTION void for_n_impl_mic(F &&f, std::integral_constant< std::size_t, 0 >) {
            f(std::integral_constant< std::size_t, 0 >());
        }

        template < typename F, std::size_t Index >
        GT_FUNCTION void for_n_impl_mic(F &&f, std::integral_constant< std::size_t, Index >) {
            for_n_impl_mic(std::forward< F >(f), std::integral_constant< std::size_t, Index - 1 >());
            f(std::integral_constant< std::size_t, Index >());
        }

        template < std::size_t N, typename F >
        GT_FUNCTION void for_n_mic(F &&f) {
            for_n_impl_mic(std::forward< F >(f), std::integral_constant< std::size_t, N - 1 >());
        }

        template < typename ExecutionEngine, typename RunFunctorArguments >
        struct run_f_on_interval_mic : run_on_block_mic< ExecutionEngine, RunFunctorArguments > {
            using base = run_on_block_mic< ExecutionEngine, RunFunctorArguments >;

            using grid_t = typename base::grid_t;
            using functor_list_t = typename base::functor_list_t;
            using extent_sizes_t = typename base::extent_sizes_t;
            using iterate_domain_t = typename base::iterate_domain_t;
            using local_domain_t = typename base::local_domain_t;
            using run_esf_functor_h_t = typename base::run_esf_functor_h_t;

            using base::run_on_block_mic;

            template < typename Interval >
            GT_FUNCTION void operator()(Interval const &) const {
                run_esf_mic< ExecutionEngine, RunFunctorArguments, Interval > run(
                    this->m_it_domain, this->m_grid, this->m_i_first, this->m_j_first, this->m_i_size, this->m_j_size);
                for_n_mic< boost::mpl::size< functor_list_t >::value >(run);
            }
        };

        template < typename ExecutionEngine, typename RunFunctorArguments >
        struct run_f_on_interval_kparallel_mic : run_on_block_mic< ExecutionEngine, RunFunctorArguments > {
            using base = run_on_block_mic< ExecutionEngine, RunFunctorArguments >;

            using grid_t = typename base::grid_t;
            using functor_list_t = typename base::functor_list_t;
            using extent_sizes_t = typename base::extent_sizes_t;
            using iterate_domain_t = typename base::iterate_domain_t;
            using local_domain_t = typename base::local_domain_t;
            using run_esf_functor_h_t = typename base::run_esf_functor_h_t;

            GT_FUNCTION
            run_f_on_interval_kparallel_mic(iterate_domain_t &it_domain,
                grid_t const &grid,
                int_t i_first,
                int_t j_first,
                int_t i_size,
                int_t j_size,
                int_t k)
                : base(it_domain, grid, i_first, j_first, i_size, j_size), m_k(k) {}

            template < typename Interval >
            GT_FUNCTION void operator()(Interval const &) const {

                typedef typename index_to_level< typename Interval::first >::type from_t;
                typedef typename index_to_level< typename Interval::second >::type to_t;

                const int_t k_first = this->m_grid.template value_at< from_t >();
                const int_t k_last = this->m_grid.template value_at< to_t >();

                if (k_first <= m_k && m_k <= k_last) {
                    run_esf_kparallel_mic< ExecutionEngine, RunFunctorArguments, Interval > run(this->m_it_domain,
                        this->m_grid,
                        this->m_i_first,
                        this->m_j_first,
                        this->m_i_size,
                        this->m_j_size,
                        m_k);

                    for_n_mic< boost::mpl::size< functor_list_t >::value >(run);
                }
            }

          private:
            int_t m_k;
        };

        template < class Left, class Right >
        struct merge_intervals {
          private:
            using left_from_t = typename index_to_level< typename Left::first >::type;
            using left_to_t = typename index_to_level< typename Left::second >::type;
            using right_from_t = typename index_to_level< typename Right::first >::type;
            using right_to_t = typename index_to_level< typename Right::second >::type;

            using from_t =
                typename boost::mpl::if_< level_lt< left_from_t, right_from_t >, left_from_t, right_from_t >::type;
            using to_t = typename boost::mpl::if_< level_gt< left_to_t, right_to_t >, left_to_t, right_to_t >::type;

          public:
            using type =
                boost::mpl::pair< typename level_to_index< from_t >::type, typename level_to_index< to_t >::type >;
        };

    } // namespace _impl

    namespace strgrid {

        template < typename RunFunctorArguments >
        class execute_kernel_functor_mic {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
            using local_domain_t = typename RunFunctorArguments::local_domain_t;
            using grid_t = typename RunFunctorArguments::grid_t;
            using reduction_data_t = typename RunFunctorArguments::reduction_data_t;
            using reduction_type_t = typename reduction_data_t::reduction_type_t;
            using loop_intervals_t = typename RunFunctorArguments::loop_intervals_t;
            using execution_type_t = typename RunFunctorArguments::execution_type_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;
            using backend_traits_t = backend_traits_from_id< enumtype::Mic >;
            using data_ptr_cached_t = typename iterate_domain_t::data_ptr_cached_t;
            using strides_cached_t = typename iterate_domain_t::strides_cached_t;

          public:
            execute_kernel_functor_mic(
                const local_domain_t &local_domain, const grid_t &grid, reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid), m_reduction_data(reduction_data) {
                char *pd = std::getenv("GT_PREFETCH_DIST");
                if (pd)
                    ::gridtools::_impl::prefetch_distance = std::atoi(pd);
                else
                    ::gridtools::_impl::prefetch_distance = 0;
            }

            template < class Execution = execution_type_t >
            typename std::enable_if< Execution::type::execution != enumtype::parallel_impl >::type operator()() {
                using namespace gridtools::_impl;

#pragma omp parallel
                {
                    iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                    data_ptr_cached_t data_ptr;
                    strides_cached_t strides;
                    int_t i_grid_size, j_grid_size, i_block_size, j_block_size, i_blocks, j_blocks;

                    init_iteration(it_domain,
                        data_ptr,
                        strides,
                        i_grid_size,
                        j_grid_size,
                        i_block_size,
                        j_block_size,
                        i_blocks,
                        j_blocks);

#pragma omp for collapse(2)
                    for (int_t bj = 0; bj < j_blocks; ++bj) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            const int_t i_first = bi * i_block_size + m_grid.i_low_bound();
                            const int_t j_first = bj * j_block_size + m_grid.j_low_bound();

                            const int_t i_bs = (bi == i_blocks - 1) ? i_grid_size - bi * i_block_size : i_block_size;
                            const int_t j_bs = (bj == j_blocks - 1) ? j_grid_size - bj * j_block_size : j_block_size;

                            run_f_on_interval_mic< execution_type_t, RunFunctorArguments > run(
                                it_domain, m_grid, i_first, j_first, i_bs, j_bs);
                            boost::mpl::for_each< loop_intervals_t >(run);
                        }
                    }

                    m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                }
                m_reduction_data.reduce();
            }

            template < class Execution = execution_type_t >
            typename std::enable_if< Execution::type::execution == enumtype::parallel_impl >::type operator()() {
                namespace mpl = boost::mpl;
                using namespace boost::mpl::placeholders;
                using namespace gridtools::_impl;

                using merged_interval_t = typename mpl::fold< loop_intervals_t,
                    typename mpl::front< loop_intervals_t >::type,
                    merge_intervals< _1, _2 > >::type;
                using from_t = typename index_to_level< typename merged_interval_t::first >::type;
                using to_t = typename index_to_level< typename merged_interval_t::second >::type;

#pragma omp parallel
                {
                    iterate_domain_t it_domain(m_local_domain, m_reduction_data.initial_value());

                    data_ptr_cached_t data_ptr;
                    strides_cached_t strides;
                    int_t i_grid_size, j_grid_size, i_block_size, j_block_size, i_blocks, j_blocks;

                    init_iteration(it_domain,
                        data_ptr,
                        strides,
                        i_grid_size,
                        j_grid_size,
                        i_block_size,
                        j_block_size,
                        i_blocks,
                        j_blocks);

                    const int_t k_first = m_grid.template value_at< from_t >();
                    const int_t k_last = m_grid.template value_at< to_t >();

#pragma omp for collapse(3)
                    for (int_t k = k_first; k <= k_last; ++k) {
                        for (int_t bj = 0; bj < j_blocks; ++bj) {
                            for (int_t bi = 0; bi < i_blocks; ++bi) {
                                const int_t i_first = bi * i_block_size + m_grid.i_low_bound();
                                const int_t j_first = bj * j_block_size + m_grid.j_low_bound();

                                const int_t i_bs =
                                    (bi == i_blocks - 1) ? i_grid_size - bi * i_block_size : i_block_size;
                                const int_t j_bs =
                                    (bj == j_blocks - 1) ? j_grid_size - bj * j_block_size : j_block_size;

                                run_f_on_interval_kparallel_mic< execution_type_t, RunFunctorArguments > run(
                                    it_domain, m_grid, i_first, j_first, i_bs, j_bs, k);
                                mpl::for_each< loop_intervals_t >(run);
                            }
                        }
                    }

                    m_reduction_data.assign(omp_get_thread_num(), it_domain.reduction_value());
                }
                m_reduction_data.reduce();
            }

          protected:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            reduction_data_t &m_reduction_data;

            void init_iteration(iterate_domain_t &it_domain,
                data_ptr_cached_t &data_ptr,
                strides_cached_t &strides,
                int_t &i_grid_size,
                int_t &j_grid_size,
                int_t &i_block_size,
                int_t &j_block_size,
                int_t &i_blocks,
                int_t &j_blocks) const {
                it_domain.set_data_pointer_impl(&data_ptr);
                it_domain.set_strides_pointer_impl(&strides);

                it_domain.template assign_storage_pointers< backend_traits_t >();
                it_domain.template assign_stride_pointers< backend_traits_t, strides_cached_t >();

                std::tie(i_block_size, j_block_size) = grid_traits_arch< enumtype::Mic >::block_size_mic(m_grid);

                i_grid_size = m_grid.i_high_bound() - m_grid.i_low_bound() + 1;
                j_grid_size = m_grid.j_high_bound() - m_grid.j_low_bound() + 1;

                i_blocks = (i_grid_size + i_block_size - 1) / i_block_size;
                j_blocks = (j_grid_size + j_block_size - 1) / j_block_size;
            }
        };

    } // namespace strgrid
} // namespace gridtools
