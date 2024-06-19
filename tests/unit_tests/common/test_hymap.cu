/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/hymap.hpp>

namespace gridtools {
    __device__ void test_compilation_issue_1715() {
        auto foo = [](auto &&i) -> auto && {
            hymap::keys<int>::make_values(i);
            return i;
        };
        foo(0);
    }
    __device__ void test_comp2() {
        int a, b, c, d;
        auto foo = [](auto const &a, auto const &b, auto &c, auto &d) {
            // GT_META_PRINT_TYPE(decltype(a));
            auto bar = hymap::keys<int, float>::template values<int const &, int const &>(a, b);

            // GT_META_PRINT_TYPE(decltype(bar));
            auto baz = hymap::keys<int, float>::template values<int &, int &>(c, d);
            GT_META_PRINT_TYPE(decltype(baz));

            baz = bar;
        };
        foo(a, b, c, d);
    }
} // namespace gridtools
