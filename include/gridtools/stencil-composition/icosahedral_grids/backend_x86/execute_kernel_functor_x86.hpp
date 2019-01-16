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

#include "../../../meta.hpp"
#include "../../backend_x86/basic_token_execution_x86.hpp"
#include "../../grid_traits_fwd.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../esf_metafunctions.hpp"
#include "../grid_traits.hpp"
#include "../stage.hpp"
#include "./iterate_domain_x86.hpp"
#include "./run_esf_functor_x86.hpp"

namespace gridtools {

    namespace icgrid {

        namespace _impl {
            template <size_t Color>
            struct loop_interval_contains_color {
                template <class T>
                GT_META_DEFINE_ALIAS(apply,
                    meta::any_of,
                    (stage_group_contains_color<Color>::template apply, GT_META_CALL(meta::at_c, (T, 2))));
            };
        } // namespace _impl

        /**
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         * @tparam IterateDomain iterator domain class
         * @tparam Grid grid object as it provided by user.
         */
        template <typename RunFunctorArguments, typename IterateDomain, typename Grid>
        struct color_execution_functor {
          private:
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

            template <class Color>
            GT_META_DEFINE_ALIAS(has_color,
                meta::any_of,
                (_impl::loop_interval_contains_color<Color::value>::template apply,
                    typename RunFunctorArguments::loop_intervals_t));

            IterateDomain &m_it_domain;
            Grid const &m_grid;
            uint_t m_loop_size;

          public:
            color_execution_functor(IterateDomain &it_domain, Grid const &grid, uint_t loop_size)
                : m_it_domain(it_domain), m_grid(grid), m_loop_size(loop_size) {}

            template <class Color, enable_if_t<has_color<Color>::value, int> = 0>
            void operator()(Color) const {
                for (uint_t j = 0; j != m_loop_size; ++j) {
                    auto memorized_index = m_it_domain.index();
                    run_functors_on_interval<RunFunctorArguments, run_esf_functor_x86<Color::value>>(
                        m_it_domain, m_grid);
                    m_it_domain.set_index(memorized_index);
                    m_it_domain.increment_j();
                }
                m_it_domain.increment_j(-m_loop_size);
                m_it_domain.increment_c();
            }
            template <class Color, enable_if_t<!has_color<Color>::value, int> = 0>
            void operator()(Color) const {
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
        struct execute_kernel_functor_x86 {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::esf_sequence_t esf_sequence_t;

            typedef typename extract_esf_location_type<esf_sequence_t>::type location_type_t;

            static constexpr int_t n_colors = location_type_t::n_colors::value;

            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            // in the x86 backend there should be only one esf per mss
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

            using iterate_domain_t = iterate_domain_x86<iterate_domain_arguments_t>;

            typedef backend_traits_from_id<target::x86> backend_traits_t;
            typedef typename iterate_domain_t::strides_cached_t strides_t;
            using interval_t = GT_META_CALL(meta::first, typename RunFunctorArguments::loop_intervals_t);
            using from_t = GT_META_CALL(meta::first, interval_t);

            execute_kernel_functor_x86(const local_domain_t &local_domain,
                const grid_t &grid,
                uint_t block_size_i,
                uint_t block_size_j,
                uint_t block_no_i,
                uint_t block_no_j)
                : m_local_domain(local_domain),
                  m_grid(grid), m_size{block_size_i + extent_t::iplus::value - extent_t::iminus::value,
                                    block_size_j + extent_t::jplus::value - extent_t::jminus::value},
                  m_block_no{block_no_i, block_no_j} {}

            void operator()() const {
                strides_t strides;

                iterate_domain_t it_domain(m_local_domain);

                it_domain.set_strides_pointer(&strides);

                it_domain.template assign_stride_pointers<backend_traits_t>();

                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    m_block_no,
                    {extent_t::iminus::value,
                        extent_t::jminus::value,
                        static_cast<int_t>(m_grid.template value_at<from_t>() - m_grid.k_min())});

                for (uint_t i = 0; i != m_size.i; ++i) {
                    gridtools::for_each<GT_META_CALL(meta::make_indices_c, n_colors)>(
                        color_execution_functor<RunFunctorArguments, iterate_domain_t, grid_t>{
                            it_domain, m_grid, m_size.j});
                    it_domain.template increment_c<-n_colors>();
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
