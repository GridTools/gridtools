/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/storage/sid.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {
        namespace tu = tuple_util;
        using tuple_util::get;

        const auto builder = storage::builder<storage_traits_t>.type<float_type>().layout<1, -1, 2, 0>();

        template <int_t I>
        using dim = integral_constant<int_t, I>;

        TEST(storage_sid, smoke) {
            auto testee = builder.dimensions(10, 20, 30, 40)();
            using testee_t = decltype(testee);

            static_assert(sid::concept_impl_::is_sid<testee_t>(), "");
            static_assert(std::is_same<sid::ptr_type<testee_t>, float_type *>(), "");
            static_assert(std::is_same<sid::ptr_diff_type<testee_t>, int_t>(), "");
            static_assert(std::is_same<sid::strides_kind<testee_t>, typename testee_t::element_type::kind_t>(), "");

            using strides_t = sid::strides_type<testee_t>;

            static_assert(tu::size<strides_t>() == 4, "");
            static_assert(std::is_same<tu::element<0, strides_t>, int_t>(), "");
            static_assert(std::is_same<tu::element<1, strides_t>, integral_constant<int_t, 0>>(), "");
            static_assert(std::is_same<tu::element<2, strides_t>, integral_constant<int_t, 1>>(), "");
            static_assert(std::is_same<tu::element<3, strides_t>, int_t>(), "");

            EXPECT_EQ(testee->get_target_ptr(), sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            auto &&expected_strides = testee->strides();

            EXPECT_EQ(expected_strides[0], get<0>(strides));
            EXPECT_EQ(expected_strides[1], get<1>(strides));
            EXPECT_EQ(expected_strides[2], get<2>(strides));
            EXPECT_EQ(expected_strides[3], get<3>(strides));

            auto lower_bounds = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, at_key<dim<0>>(lower_bounds));
            EXPECT_EQ(0, at_key<dim<2>>(lower_bounds));
            EXPECT_EQ(0, at_key<dim<3>>(lower_bounds));

            auto &&lengths = testee->lengths();
            auto upper_bounds = sid::get_upper_bounds(testee);
            EXPECT_EQ(lengths[0], at_key<dim<0>>(upper_bounds));
            EXPECT_EQ(lengths[2], at_key<dim<2>>(upper_bounds));
            EXPECT_EQ(lengths[3], at_key<dim<3>>(upper_bounds));
        }

        TEST(storage_sid, regression_strides_of_small_storage) {
            auto testee = builder.dimensions(1, 1, 1, 1)();

            auto strides = sid::get_strides(testee);
            auto expected_strides = testee->strides();

            EXPECT_EQ(expected_strides[0], get<0>(strides));
            EXPECT_EQ(expected_strides[1], get<1>(strides));
            EXPECT_EQ(expected_strides[2], get<2>(strides));
            EXPECT_EQ(expected_strides[3], get<3>(strides));
        }

        TEST(storage_sid, scalar) {
            auto testee = storage::builder<storage_traits_t>.type<float_type>().layout<-1>().dimensions(10)();
            using testee_t = decltype(testee);

            static_assert(sid::concept_impl_::is_sid<testee_t>(), "");

            using diff_t = sid::ptr_diff_type<testee_t>;
            static_assert(std::is_empty<diff_t>(), "");

            auto ptr = sid::get_origin(testee)();

            EXPECT_EQ(ptr, ptr + diff_t{});

            using lower_bounds_t = sid::lower_bounds_type<testee_t>;
            using upper_bounds_t = sid::upper_bounds_type<testee_t>;

            static_assert(tuple_util::size<lower_bounds_t>() == 0, "");
            static_assert(tuple_util::size<upper_bounds_t>() == 0, "");
        }
    } // namespace
} // namespace gridtools
