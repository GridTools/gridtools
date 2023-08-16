/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/storage/adapter/nanobind_adapter.hpp>

namespace nb = nanobind;

TEST(NanobindAdapter, DataDynStrides) {
    const auto data = reinterpret_cast<void *>(0xDEADBEEF);
    constexpr int ndim = 2;
    constexpr std::array<std::size_t, ndim> shape = {3, 4};
    constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
    nb::ndarray<int, nb::shape<nb::any, nb::any>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

    const auto sid = gridtools::nanobind::as_sid(ndarray);
    const auto s_origin = sid_get_origin(sid);
    const auto s_strides = sid_get_strides(sid);
    const auto s_ptr = s_origin();

    EXPECT_EQ(s_ptr, data);
    EXPECT_EQ(strides[0], gridtools::get<0>(s_strides));
    EXPECT_EQ(strides[1], gridtools::get<1>(s_strides));
}

TEST(NanobindAdapter, StaticStridesMatch) {
    const auto data = reinterpret_cast<void *>(0xDEADBEEF);
    constexpr int ndim = 2;
    constexpr std::array<std::size_t, ndim> shape = {3, 4};
    constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
    nb::ndarray<int, nb::shape<nb::any, nb::any>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

    const auto sid = gridtools::nanobind::as_sid(ndarray, gridtools::nanobind::stride_spec<1, nanobind::any>{});
    const auto s_strides = sid_get_strides(sid);

    EXPECT_EQ(strides[0], gridtools::get<0>(s_strides).value);
    EXPECT_EQ(strides[1], gridtools::get<1>(s_strides));
}

TEST(NanobindAdapter, StaticStridesMismatch) {
    const auto data = reinterpret_cast<void *>(0xDEADBEEF);
    constexpr int ndim = 2;
    constexpr std::array<std::size_t, ndim> shape = {3, 4};
    constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
    nb::ndarray<int, nb::shape<nb::any, nb::any>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

    EXPECT_THROW(
        gridtools::nanobind::as_sid(ndarray, gridtools::nanobind::stride_spec<2, nanobind::any>{}), std::invalid_argument);
}
