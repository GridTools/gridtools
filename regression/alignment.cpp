/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
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
