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
#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/storage/storage-facility.hpp>

#include "backend_select.hpp"

using storage_traits_t = typename backend_t::storage_traits_t;
using storage_info_t = storage_traits_t::storage_info_t<0, 3>;
using data_store_t = storage_traits_t::data_store_t<gridtools::float_type, storage_info_t>;

struct my_type {
    int value;

    inline bool operator==(my_type const &other) const { return value == other.value; }
};

TEST(global_accessor_simple, arg_storage_pair) {
    auto test_val = my_type{123};
    using p_global_param = gridtools::arg<0, my_type>;

    auto pair = (p_global_param{} = test_val);

    ASSERT_EQ(test_val, pair.m_value.m_value);
}

struct functor1 {
    using out = gridtools::inout_accessor<0>;
    using global_acc = gridtools::global_accessor<1>;

    typedef boost::mpl::vector<out, global_acc> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        auto global_acc_value = eval(global_acc());
        eval(out()) = global_acc_value.value;
    }
};

TEST(global_accessor_simple, make_computation) {
    auto test_val = my_type{123};

    storage_info_t sinfo{10, 10, 10};
    data_store_t out{sinfo, -1.};

    using p_out = gridtools::arg<0, data_store_t>;
    using p_global_param = gridtools::arg<1, my_type>;

    auto grid = gridtools::make_grid(10, 10, 10);

    auto stencil = gridtools::make_computation<backend_t>(grid,
        p_out{} = out,
        p_global_param{} = test_val,
        gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
            gridtools::make_stage<functor1>(p_out{}, p_global_param{})));

    stencil.run();

    out.sync();
    auto outv = gridtools::make_host_view(out);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                ASSERT_EQ(test_val.value, outv(i, j, k));
            }
        }
    }
}

// TEST(global_accessor_simple, test) {
//    gridtools::float_type test_value = 3.1415;
//    storage_info_t sinfo{10, 10, 10};
//    data_store_t out{sinfo, -1.};
//
//    auto global_param_value = backend_t::make_global_parameter(test_value);
//
//    using p_out = gridtools::arg<0, data_store_t>;
//    using p_global_param = gridtools::arg<1, decltype(global_param_value)>;
//    //    using p_global_param = gridtools::arg<1, gridtools::float_type>;
//
//    auto grid = gridtools::make_grid(10, 10, 10);
//
//    auto stencil = gridtools::make_computation<backend_t>(grid,
//        p_out() = out,
//        p_global_param{} = global_param_value,
//        gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
//            gridtools::make_stage<functor1>(p_out{}, p_global_param{})));
//
//    stencil.run();
//
//    out.sync();
//
//    auto outv = gridtools::make_host_view(out);
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            for (int k = 0; k < 10; ++k) {
//                ASSERT_EQ(test_value, outv(i, j, k));
//            }
//        }
//    }
//}
