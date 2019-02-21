/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "../../cuda_gtest_plugin.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace gridtools::execute;
using namespace gridtools::expressions;

/*
 * Mocking accessor and iterate domain
 */
template <typename T>
struct accessor_mock {
    using return_type = T;
    T value;
};

namespace gridtools {
    template <typename T>
    struct is_accessor<accessor_mock<T>> : boost::mpl::true_ {};
} // namespace gridtools

struct iterate_domain_mock {
    template <typename... Ts>
    GT_FUNCTION iterate_domain_mock(Ts...) {}

    // trivial evaluation of the accessor_mock
    template <typename Accessor, typename std::enable_if<is_accessor<Accessor>::value, int>::type = 0>
    GT_FUNCTION typename Accessor::return_type operator()(Accessor const &val) const {
        return val.value;
    }

    // copy of the iterate_domain for expr
    template <class Op, class... Args>
    GT_FUNCTION auto operator()(expr<Op, Args...> const &arg) const
        GT_AUTO_RETURN(expressions::evaluation::value(*this, arg));
};

using val = accessor_mock<float>;

/*
 * User API tests
 */
CUDA_TEST(test_expressions, add_accessors) {
    iterate_domain_mock eval(0);

    auto add = val{1} + val{2};

    auto result = eval(add);
    ASSERT_FLOAT_EQ(result, (float)3.);
}

CUDA_TEST(test_expressions, sub_accessors) {
    iterate_domain_mock eval(0);

    auto sub = val{1} - val{2};

    auto result = eval(sub);
    ASSERT_FLOAT_EQ(result, (float)-1.);
}

CUDA_TEST(test_expressions, negate_accessor) {
    iterate_domain_mock eval(0);

    auto negate = -val{1};

    auto result = eval(negate);
    ASSERT_FLOAT_EQ(result, (float)-1.);
}

CUDA_TEST(test_expressions, plus_sign_accessor) {
    iterate_domain_mock eval(0);

    auto negate = +val{1};

    auto result = eval(negate);
    ASSERT_FLOAT_EQ(result, (float)1.);
}

CUDA_TEST(test_expressions, with_parenthesis) {
    iterate_domain_mock eval(0);

    auto expression = (val{1} + val{2}) * (val{1} - val{2});

    auto result = eval(expression);
    ASSERT_FLOAT_EQ(result, (float)-3.);
}

/*
 * Library tests illustrating how expressions are evaluated
 */
TEST(test_expressions, expr_plus_by_ctor) {
    accessor_mock<double> a{1};
    accessor_mock<double> b{2};

    auto add = make_expr(plus_f{}, a, b);

    iterate_domain_mock iterate;

    auto result = evaluation::value(iterate, add);

    ASSERT_DOUBLE_EQ(result, 3);
}

TEST(expression_unit_test, expr_plus_by_operator_overload) {
    iterate_domain_mock iterate;

    auto add = val{1} + val{2};

    auto result = evaluation::value(iterate, add);

    ASSERT_DOUBLE_EQ(result, 3);
}
