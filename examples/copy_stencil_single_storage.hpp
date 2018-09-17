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

#include "backend_select.hpp"
#include <gridtools/stencil-composition/stencil-composition.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;

using namespace gridtools;
using namespace enumtype;

namespace copy_stencil {
    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef accessor<0, enumtype::inout, extent<0, 0, 0, 0>, 5> in;
        typedef boost::mpl::vector<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(in()) = eval(in(dimension<5>(1)));
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        typedef storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
        typedef storage_traits<backend_t::backend_id_t>::data_store_field_t<float_type, storage_info_t, 2>
            data_store_field_t;
        storage_info_t meta_data_(x, y, z);

        // Definition of the actual data fields that are used for input/output
        data_store_field_t in(meta_data_);
        auto inv = make_field_host_view(in);
        auto inv00 = inv.get<0, 0>();
        auto inv01 = inv.get<0, 1>();
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    inv00(i, j, k) = i + j + k;
                    inv01(i, j, k) = 0.;
                }
            }
        }

        typedef arg<0, data_store_field_t> p_in;

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        halo_descriptor di{0, 0, 0, d1 - 1, d1};
        halo_descriptor dj{0, 0, 0, d2 - 1, d2};

        auto grid = make_grid(d1, d2, d3);

        auto copy = gridtools::make_computation<backend_t>(grid,
            p_in() = in,
            gridtools::make_multistage(execute<forward>(), gridtools::make_stage<copy_functor>(p_in())));

        copy.run();

        copy.sync_bound_data_stores();

#ifdef BENCHMARK
        std::cout << copy.print_meter() << std::endl;
#endif

        in.sync();
        bool success = true;
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    if (inv00(i, j, k) != inv01(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << (inv00(i, j, k)) << ", out = " << (inv01(i, j, k)) << std::endl;
                        success = false;
                    }
                }
            }
        }
        return success;
    }
} // namespace copy_stencil
