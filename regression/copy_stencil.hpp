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
#include "benchmarker.hpp"
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

        typedef accessor<0, enumtype::in, extent<>, 3> in;
        typedef accessor<1, enumtype::inout, extent<>, 3> out;
        typedef boost::mpl::vector<in, out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    struct copy_stencil_test {
        uint d1, d2, d3, t_steps;
        bool m_verify;

        typedef storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
        typedef storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t> data_store_t;
        storage_info_t meta_data_;
        data_store_t in, out;

        typedef arg<0, data_store_t> p_in;
        typedef arg<1, data_store_t> p_out;

        gridtools::grid<gridtools::axis<1>::axis_interval_t> grid;
        // gridtools::grid< axis > grid;
        copy_stencil_test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool m_verify)
            : d1(x), d2(y), d3(z), t_steps(t_steps), m_verify(m_verify), meta_data_(x, y, z),
              in(meta_data_, [](int i, int j, int k) { return i + j + k; }, "in"), out(meta_data_, -1.0, "out"),
              grid(halo_descriptor(d1),
                  halo_descriptor(d2),
                  _impl::intervals_to_indices(gridtools::axis<1>{d3}.interval_sizes()))

        {}

        bool verify() {
            auto in_v = make_host_view(in);
            auto out_v = make_host_view(out);
            // check consistency
            assert(check_consistency(in, in_v) && "view cannot be used safely.");
            assert(check_consistency(out, out_v) && "view cannot be used safely.");

            bool success = true;
            if (m_verify) {
                for (uint_t i = 0; i < d1; ++i) {
                    for (uint_t j = 0; j < d2; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            if (in_v(i, j, k) != out_v(i, j, k)) { // TODO use verifier
                                std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                          << "in = " << in_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                                success = false;
                            }
                        }
                    }
                }
            }
            return success;
        }

        bool test_with_extents() {
            auto copy = gridtools::make_computation<backend_t>(grid,
                p_in() = in,
                p_out() = out,
                gridtools::make_multistage // mss_descriptor
                (execute<parallel, 20>(),
                    gridtools::make_stage_with_extent<copy_functor, extent<0, 0, 0, 0>>(p_in(), p_out())));

            copy.run();

            out.sync();
            in.sync();

            return verify();
        }

        bool test() {
            auto copy = gridtools::make_computation<backend_t>(grid,
                p_in() = in,
                p_out() = out,
                gridtools::make_multistage // mss_descriptor
                (execute<parallel>(), gridtools::make_stage<copy_functor>(p_in(), p_out())));

            copy.run();

            out.sync();
            in.sync();

            bool success = verify();

#ifdef BENCHMARK
            benchmarker::run(copy, t_steps);
#endif
            return success;
        }
    };
} // namespace copy_stencil
