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

#include <utility>

#include "execute_kernel_functor_mic_fwd.hpp"
#include "../../run_functor_arguments_fwd.hpp"

namespace gridtools {

    namespace strgrid {
        template <>
        struct grid_traits_arch< enumtype::Mic > {
            template < typename RunFunctorArguments >
            struct kernel_functor_executor {
                GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
                typedef execute_kernel_functor_mic< RunFunctorArguments > type;
            };

            template < typename Grid >
            static std::pair< int_t, int_t > block_size_mic(Grid const &grid) {
                const int_t i_grid_size = grid.i_high_bound() - grid.i_low_bound() + 1;
                const int_t j_grid_size = grid.j_high_bound() - grid.j_low_bound() + 1;

                const int_t threads = omp_get_max_threads();

                const int_t j_block_size = (j_grid_size + threads - 1) / threads;
                const int_t j_blocks = (j_grid_size + j_block_size - 1) / j_block_size;
                const int_t i_blocks = threads / j_blocks;
                const int_t i_block_size = (i_grid_size + i_blocks - 1) / i_blocks;

                // currently required by the implementation
                assert(i_blocks * j_blocks <= threads);

                return std::make_pair(i_block_size, j_block_size);
            }
        };
    }
}
