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

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/macros.hpp>
#include <gridtools/meta/type_traits.hpp>
#include <gridtools/stencil-composition/sid/concept.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {
        using traits_t = storage_traits<target_t>;
        using storage_info_t = traits_t::custom_layout_storage_info_t<0, layout_map<1, -1, 2, 0>>;
        using data_store_t = traits_t::data_store_t<float_type, storage_info_t>;
        namespace tu = tuple_util;
        using tuple_util::get;

        TEST(storage_sid, smoke) {
            static_assert(is_sid<data_store_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, data_store_t), float_type *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, data_store_t), int_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, data_store_t), storage_info_t>(), "");

            using strides_t = GT_META_CALL(sid::strides_type, data_store_t);

            static_assert(tu::size<strides_t>() == 4, "");
            static_assert(std::is_same<GT_META_CALL(tu::element, (0, strides_t)), int_t>(), "");
            static_assert(std::is_same<GT_META_CALL(tu::element, (1, strides_t)), integral_constant<int_t, 0>>(), "");
            static_assert(std::is_same<GT_META_CALL(tu::element, (2, strides_t)), integral_constant<int_t, 1>>(), "");
            static_assert(std::is_same<GT_META_CALL(tu::element, (3, strides_t)), int_t>(), "");

            data_store_t testee = {{10, 10, 10, 10}, 0};

            EXPECT_EQ(advanced_get_raw_pointer_of(make_target_view(testee)), sid::get_origin(testee));

            auto strides = sid::get_strides(testee);
            auto expected_strides = testee.strides();

            EXPECT_EQ(expected_strides[0], get<0>(strides));
            EXPECT_EQ(expected_strides[1], get<1>(strides));
            EXPECT_EQ(expected_strides[2], get<2>(strides));
            EXPECT_EQ(expected_strides[3], get<3>(strides));
        }

        TEST(storage_sid, as_host) {
            data_store_t data = {{10, 10, 10, 10}, 42};
            auto testee = as_host(data);

            using testee_t = decltype(testee);

            static_assert(is_sid<testee_t>(), "");
            static_assert(
                std::is_same<GT_META_CALL(sid::ptr_type, testee_t), GT_META_CALL(sid::ptr_type, data_store_t)>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee_t),
                              GT_META_CALL(sid::ptr_diff_type, data_store_t)>(),
                "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee_t),
                              GT_META_CALL(sid::strides_kind, data_store_t)>(),
                "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_type, testee_t),
                              GT_META_CALL(sid::strides_type, data_store_t)>(),
                "");

            auto testee_strides = sid::get_strides(testee);
            auto data_strides = sid::get_strides(data);

            EXPECT_EQ(get<0>(data_strides), get<0>(testee_strides));
            EXPECT_EQ(get<1>(data_strides), get<1>(testee_strides));
            EXPECT_EQ(get<2>(data_strides), get<2>(testee_strides));
            EXPECT_EQ(get<3>(data_strides), get<3>(testee_strides));

            // we can dereference in the host context
            EXPECT_EQ(42, *sid::get_origin(testee));
        }

        TEST(storage_sid, scalar) {
            using storage_info_t = traits_t::custom_layout_storage_info_t<0, layout_map<-1>>;
            using testee_t = traits_t::data_store_t<float_type, storage_info_t>;

            static_assert(sid::concept_impl_::is_sid<testee_t>(), "");

            using diff_t = GT_META_CALL(sid::ptr_diff_type, testee_t);
            static_assert(std::is_empty<diff_t>(), "");

            testee_t testee = {storage_info_t{10}, 0};

            auto ptr = sid::get_origin(testee);

            EXPECT_EQ(ptr, ptr + diff_t{});
        }
    } // namespace
} // namespace gridtools
