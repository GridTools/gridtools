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

#include "backend_select.hpp"
#include <gridtools/stencil-composition/stencil-composition.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/
namespace gt = gridtools;

// These are the stencil operators that compose the multistage stencil in this test
struct copy_functor {
    using in = gt::accessor<0, gt::enumtype::in, gt::extent<>, 3>;
    using out = gt::accessor<1, gt::enumtype::inout, gt::extent<>, 3>;
    typedef gt::make_arg_list<in, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in());
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

void handle_error(gt::int_t) { std::cout << "error" << std::endl; }

template <typename In, typename Out>
bool verify(In const& in, Out const& out) {
    auto in_v = gt::make_host_view(in);
    auto out_v = gt::make_host_view(out);
    // check consistency
    assert(gt::check_consistency(in, in_v) && "view cannot be used safely.");
    assert(gt::check_consistency(out, out_v) && "view cannot be used safely.");

    bool success = true;
    for (int k = in_v.template total_begin<2>(); k <= in_v.template total_end<2>(); ++k) {
        for (int i = in_v.template total_begin<0>(); i <= in_v.template total_end<0>(); ++i) {
            for (int j = in_v.template total_begin<1>(); j <= in_v.template total_end<1>(); ++j) {
                if (in_v(i, j, k) != out_v(i, j, k)) { // TODO use verifier
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "in = " << in_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    return success;
}

int main(int argc, char** argv) {

    gt::uint_t d1, d2, d3;
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    typedef gt::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
    typedef gt::storage_traits<backend_t::backend_id_t>::data_store_t<gt::float_type, storage_info_t> data_store_t;
    storage_info_t meta_data_(d1,d2,d3);

    typedef gt::arg<0, data_store_t> p_in;
    typedef gt::arg<1, data_store_t> p_out;

    gt::grid<gt::axis<1>::axis_interval_t> grid(gt::halo_descriptor(d1),
                                                gt::halo_descriptor(d2),
                                                gt::_impl::intervals_to_indices(gt::axis<1>{d3}.interval_sizes()));

    data_store_t in(meta_data_, [](int i, int j, int k) { return i + j + k; }, "in");
    data_store_t out(meta_data_, -1.0, "out");

    auto copy = gt::make_computation<backend_t>
        (grid,
         p_in() = in,
         p_out() = out,
         gt::make_multistage // mss_descriptor
         (gt::enumtype::execute<gt::enumtype::parallel>(), gt::make_stage<copy_functor>(p_in(), p_out())));

    copy.run();

    out.sync();
    in.sync();

    bool success = verify(in, out);

    if (success) {
        std::cout << "Successful\n";
    } else {
        std::cout << "Failed\n";
    }

    return !success;
};
