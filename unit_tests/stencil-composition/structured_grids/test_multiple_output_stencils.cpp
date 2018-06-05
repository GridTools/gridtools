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
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

struct TensionShearFunction {
    using T_sqr_s = inout_accessor< 0 >;
    using S_sqr_uv = inout_accessor< 1 >;

    using u_in = in_accessor< 2, extent< -1, 0, 0, 1 > >;
    using v_in = in_accessor< 3, extent< 0, 1, -1, 0 > >;

    using arg_list = boost::mpl::vector< T_sqr_s, S_sqr_uv, u_in, v_in >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {}
};

struct SmagCoeffFunction {
    using smag_u = inout_accessor< 0 >;
    using smag_v = inout_accessor< 1 >;

    using T_sqr_s = in_accessor< 2, extent< 0, 1, 0, 1 > >;
    using S_sqr_uv = in_accessor< 3, extent< -1, 0, -1, 0 > >;

    using arg_list = boost::mpl::vector< smag_u, smag_v, T_sqr_s, S_sqr_uv >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {}
};

struct SmagUpdateFunction {
    using u_out = inout_accessor< 0 >;
    using v_out = inout_accessor< 1 >;

    using u_in = in_accessor< 2, extent< -1, 1, -1, 1 > >;
    using v_in = in_accessor< 3, extent< -1, 1, -1, 1 > >;
    using smag_u = in_accessor< 4 >;
    using smag_v = in_accessor< 5 >;

    using arg_list = boost::mpl::vector< u_out, v_out, u_in, v_in, smag_u, smag_v >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {}
};

TEST(multiple_outputs, compute_extents) {

    typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

    using T_sqr_s = tmp_arg< 0, storage_t >;
    using S_sqr_uv = tmp_arg< 1, storage_t >;
    using smag_u = tmp_arg< 2, storage_t >;
    using smag_v = tmp_arg< 3, storage_t >;

    // Output fields
    using u_out = arg< 4, storage_t >;
    using v_out = arg< 5, storage_t >;

    // Input fields
    using u_in = arg< 6, storage_t >;
    using v_in = arg< 7, storage_t >;

    halo_descriptor di{2, 2, 2, 7, 10};
    halo_descriptor dj{2, 2, 2, 7, 10};
    auto grid_ = make_grid(di, dj, 10);

    make_computation< backend_t >(
        grid_,
        make_multistage(execute< forward >(),
            make_stage< TensionShearFunction >(T_sqr_s(), S_sqr_uv(), u_in(), v_in()),
            make_stage< SmagCoeffFunction >(smag_u(), smag_v(), T_sqr_s(), S_sqr_uv()),
            make_stage< SmagUpdateFunction >(u_out(), v_out(), u_in(), v_in(), smag_u(), smag_v())));
}
