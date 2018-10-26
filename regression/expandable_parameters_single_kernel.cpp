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
#undef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct functor_single_kernel {
    using parameters1_out = inout_accessor<0>;
    using parameters2_out = inout_accessor<1>;
    using parameters3_out = inout_accessor<2>;
    using parameters4_out = inout_accessor<3>;
    using parameters5_out = inout_accessor<4>;

    using parameters1_in = in_accessor<5>;
    using parameters2_in = in_accessor<6>;
    using parameters3_in = in_accessor<7>;
    using parameters4_in = in_accessor<8>;
    using parameters5_in = in_accessor<9>;

    using arg_list = boost::mpl::vector<parameters1_out,
        parameters2_out,
        parameters3_out,
        parameters4_out,
        parameters5_out,
        parameters1_in,
        parameters2_in,
        parameters3_in,
        parameters4_in,
        parameters5_in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(parameters1_out()) = eval(parameters1_in());
        eval(parameters2_out()) = eval(parameters2_in());
        eval(parameters3_out()) = eval(parameters3_in());
        eval(parameters4_out()) = eval(parameters4_in());
        eval(parameters5_out()) = eval(parameters5_in());
    }
};

using ExpandableParameters = regression_fixture<>;

TEST_F(ExpandableParameters, Test) {
    std::vector<storage_type> out = {
        make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    std::vector<storage_type> in = {
        make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    arg<0, storage_type> p_0_out;
    arg<1, storage_type> p_1_out;
    arg<2, storage_type> p_2_out;
    arg<3, storage_type> p_3_out;
    arg<4, storage_type> p_4_out;

    arg<5, storage_type> p_0_in;
    arg<6, storage_type> p_1_in;
    arg<7, storage_type> p_2_in;
    arg<8, storage_type> p_3_in;
    arg<9, storage_type> p_4_in;

    tmp_arg<10, storage_type> p_0_tmp;
    tmp_arg<11, storage_type> p_1_tmp;
    tmp_arg<12, storage_type> p_2_tmp;
    tmp_arg<13, storage_type> p_3_tmp;
    tmp_arg<14, storage_type> p_4_tmp;

    make_computation(p_0_out = out[0],
        p_1_out = out[1],
        p_2_out = out[2],
        p_3_out = out[3],
        p_4_out = out[4],
        p_0_in = in[0],
        p_1_in = in[1],
        p_2_in = in[2],
        p_3_in = in[3],
        p_4_in = in[4],
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_0_tmp, p_1_tmp, p_2_tmp, p_3_tmp, p_4_tmp)),
            make_stage<functor_single_kernel>(
                p_0_tmp, p_1_tmp, p_2_tmp, p_3_tmp, p_4_tmp, p_0_in, p_1_in, p_2_in, p_3_in, p_4_in),
            make_stage<functor_single_kernel>(
                p_0_out, p_1_out, p_2_out, p_3_out, p_4_out, p_0_tmp, p_1_tmp, p_2_tmp, p_3_tmp, p_4_tmp)))
        .run();
    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
