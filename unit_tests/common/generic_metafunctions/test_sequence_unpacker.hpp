/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/generic_metafunctions/sequence_unpacker.hpp>

#include <type_traits>

#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/generic_metafunctions/variadic_typedef.hpp>

using namespace gridtools;

void test_sequence_unpacker(bool *result) {
    *result = true;

    using test_type = boost::mpl::vector4<int, float, char, double>;

    static_assert(
        std::is_same<sequence_unpacker<test_type>::type, variadic_typedef<int, float, char, double>>::value, "Error");
}
