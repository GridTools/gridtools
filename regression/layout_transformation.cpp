/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/interface/layout_transformation/layout_transformation.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/cartesian_regression_fixture.hpp>

#include <gtest/gtest.h>

using namespace gridtools;

template <typename Src, typename Dst>
void verify_result(Src &src, Dst &dst) {
    auto src_v = src->const_host_view();
    auto dst_v = dst->const_host_view();

    auto &&lengths = src->lengths();
    for (int i = 0; i < lengths[0]; ++i)
        for (int j = 0; j < lengths[1]; ++j)
            for (int k = 0; k < lengths[2]; ++k)
                EXPECT_EQ(src_v(i, j, k), dst_v(i, j, k));
}

using layout_transformation = cartesian::regression_fixture<>;

TEST_F(layout_transformation, ijk_to_kji) {
    auto src = builder().layout<0, 1, 2>().initializer([](int i, int j, int k) { return i + j + k; })();
    auto dst = builder().layout<2, 1, 0>()();
    auto testee = [&] {
        interface::transform(
            dst->get_target_ptr(), src->get_target_ptr(), src->lengths(), dst->strides(), src->strides());
    };
    testee();
    verify_result(src, dst);
    benchmark(testee);
}
