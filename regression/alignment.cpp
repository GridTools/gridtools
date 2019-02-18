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

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

/**
  @file
  This file shows an implementation of the stencil in which a misaligned storage is aligned
*/

using namespace gridtools;

using alignment_test = regression_fixture<2>;

struct not_aligned {
    using acc = inout_accessor<0>;
    using out = inout_accessor<1>;
    using param_list = make_param_list<acc, out>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        auto *ptr = &eval(acc{});
        constexpr auto alignment = sizeof(decltype(*ptr)) * alignment_test::storage_info_t::alignment_t::value;
        constexpr auto halo_size = alignment_test::halo_size;
        eval(out{}) = eval.i() == halo_size && reinterpret_cast<ptrdiff_t>(ptr) % alignment;
    }
};

TEST_F(alignment_test, test) {
    using bool_storage = storage_tr::data_store_t<bool, storage_info_t>;
    arg<0, bool_storage> p_out;
    auto out = make_storage<bool_storage>();
    make_positional_computation<backend_t>(make_grid(),
        p_0 = make_storage(),
        p_out = out,
        make_multistage(execute::forward(), make_stage<not_aligned>(p_0, p_out)))
        .run();
    verify(make_storage<bool_storage>(false), out);
}
