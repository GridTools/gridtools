/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/storage/adapter/nanobind_adapter.hpp>

#include <Python.h>
#include <array>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>

#include <gtest/gtest.h>

namespace nb = nanobind;

class python_init_fixture : public ::testing::Test {
  protected:
    void SetUp() override { Py_Initialize(); }
    void TearDown() override { Py_FinalizeEx(); }
};

namespace gridtools {
    TEST_F(python_init_fixture, NanobindAdapterDataDynStrides) {
        const auto data = reinterpret_cast<void *>(0xDEADBEEF);
        constexpr int ndim = 2;
        constexpr std::array<std::size_t, ndim> shape = {3, 4};
        constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
        nb::ndarray<int, nb::shape<-1, -1>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

        const auto sid = gridtools::nanobind::as_sid(ndarray);
        const auto s_origin = sid::get_origin(sid);
        const auto s_strides = sid::get_strides(sid);
        const auto s_ptr = s_origin();
        const auto s_lower_bound = sid::get_lower_bounds(sid);
        const auto s_upper_bound = sid::get_upper_bounds(sid);

        EXPECT_EQ(s_ptr, data);
        EXPECT_EQ(strides[0], gridtools::get<0>(s_strides));
        EXPECT_EQ(strides[1], gridtools::get<1>(s_strides));

        EXPECT_EQ(0, gridtools::get<0>(s_lower_bound));
        EXPECT_EQ(0, gridtools::get<1>(s_lower_bound));
        EXPECT_EQ(3, gridtools::get<0>(s_upper_bound));
        EXPECT_EQ(4, gridtools::get<1>(s_upper_bound));
    }

    TEST_F(python_init_fixture, NanobindAdapterReadOnly) {
        const auto data = reinterpret_cast<void *>(0xDEADBEEF);
        constexpr int ndim = 2;
        constexpr std::array<std::size_t, ndim> shape = {3, 4};
        constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
        nb::ndarray<int, nb::shape<-1, -1>, nb::ro> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

        const auto sid = gridtools::nanobind::as_sid(ndarray);
        using element_t = gridtools::sid::element_type<decltype(sid)>;
        static_assert(std::is_same_v<element_t, int const>);

        const auto s_origin = sid::get_origin(sid);
        const auto s_strides = sid::get_strides(sid);
        const auto s_ptr = s_origin();

        EXPECT_EQ(s_ptr, data);
        EXPECT_EQ(strides[0], gridtools::get<0>(s_strides));
        EXPECT_EQ(strides[1], gridtools::get<1>(s_strides));
    }

    TEST_F(python_init_fixture, NanobindAdapterStaticStridesMatch) {
        const auto data = reinterpret_cast<void *>(0xDEADBEEF);
        constexpr int ndim = 2;
        constexpr std::array<std::size_t, ndim> shape = {3, 4};
        constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
        nb::ndarray<int, nb::shape<-1, -1>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

        const auto sid = gridtools::nanobind::as_sid(ndarray, gridtools::nanobind::stride_spec<1, -1>{});
        const auto s_strides = sid::get_strides(sid);

        EXPECT_EQ(strides[0], gridtools::get<0>(s_strides).value);
        EXPECT_EQ(strides[1], gridtools::get<1>(s_strides));
    }

    TEST_F(python_init_fixture, NanobindAdapterStaticStridesMismatch) {
        const auto data = reinterpret_cast<void *>(0xDEADBEEF);
        constexpr int ndim = 2;
        constexpr std::array<std::size_t, ndim> shape = {3, 4};
        constexpr std::array<std::intptr_t, ndim> strides = {1, 3};
        nb::ndarray<int, nb::shape<-1, -1>> ndarray{data, ndim, shape.data(), nb::handle{}, strides.data()};

        EXPECT_THROW(
            gridtools::nanobind::as_sid(ndarray, gridtools::nanobind::stride_spec<2, -1>{}), std::invalid_argument);
    }
} // namespace gridtools
