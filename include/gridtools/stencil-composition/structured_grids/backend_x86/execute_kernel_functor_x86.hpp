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
 * execute_kernel_functor_x86.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "../../backend_x86/basic_token_execution_x86.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../positional_iterate_domain.hpp"
#include "./iterate_domain_x86.hpp"
#include "./run_esf_functor_x86.hpp"

namespace gridtools {

    namespace strgrid {

        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_x86 {
          private:
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;

            using iterate_domain_arguments_t = iterate_domain_arguments<typename RunFunctorArguments::backend_ids_t,
                local_domain_t,
                typename RunFunctorArguments::esf_sequence_t,
                std::tuple<>,
                grid_t>;
            using iterate_domain_x86_t = iterate_domain_x86<iterate_domain_arguments_t>;
            using iterate_domain_t = typename conditional_t<local_domain_is_stateful<local_domain_t>::value,
                meta::lazy::id<positional_iterate_domain<iterate_domain_x86_t>>,
                meta::lazy::id<iterate_domain_x86_t>>::type;

            typedef backend_traits_from_id<target::x86> backend_traits_t;

            using extent_t = GT_META_CALL(
                get_extent_from_loop_intervals, typename RunFunctorArguments::loop_intervals_t);

            using interval_t = GT_META_CALL(meta::first, typename RunFunctorArguments::loop_intervals_t);
            using from_t = GT_META_CALL(meta::first, interval_t);

            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            pos3<uint_t> m_size;
            pos3<uint_t> m_block_no;

          public:
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
                iterate_domain_t it_domain(m_local_domain);

                it_domain.template assign_stride_pointers<backend_traits_t>();

                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    m_block_no,
                    {extent_t::iminus::value,
                        extent_t::jminus::value,
                        static_cast<int_t>(m_grid.template value_at<from_t>() - m_grid.k_min())});

                // run the nested ij loop
                typename iterate_domain_t::array_index_t irestore_index, jrestore_index;
                for (uint_t i = 0; i != m_size.i; ++i) {
                    irestore_index = it_domain.index();
                    for (uint_t j = 0; j != m_size.j; ++j) {
                        jrestore_index = it_domain.index();
                        run_functors_on_interval<RunFunctorArguments, run_esf_functor_x86>(it_domain, m_grid);
                        it_domain.set_index(jrestore_index);
                        it_domain.increment_j();
                    }
                    it_domain.set_index(irestore_index);
                    it_domain.increment_i();
                }
            }
        };
    } // namespace strgrid
} // namespace gridtools
