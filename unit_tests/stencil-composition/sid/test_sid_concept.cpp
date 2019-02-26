/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/sid/concept.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/functional.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        // several primitive not sids
        static_assert(!is_sid<void>(), "");
        static_assert(!is_sid<int>(), "");
        struct garbage {};
        static_assert(!is_sid<garbage>(), "");

        // fully custom defined sid
        namespace custom {
            struct element {};
            struct ptr_diff {
                int val;
            };
            struct ptr {
                element *val;
                GT_FUNCTION element &operator*() const { return *val; }
                friend GT_FUNCTION ptr operator+(ptr, ptr_diff) { return {}; }
            };
            struct stride {
                friend GT_FUNCTION std::true_type sid_shift(ptr &, stride const &, int) { return {}; }
                friend GT_FUNCTION std::false_type sid_shift(ptr_diff &, stride const &, int) { return {}; }
            };

            struct dim_0;
            struct dim_1;

            using strides = hymap::keys<dim_0, dim_1>::values<stride, stride>;

            struct strides_kind;
            struct bounds_validator_kind;

            struct testee {
                friend host_device::constant<ptr> sid_get_origin(testee &) { return {}; }
                friend strides sid_get_strides(testee const &) { return {}; }

                friend ptr_diff sid_get_ptr_diff(testee);
                friend strides_kind sid_get_strides_kind(testee);
                friend bounds_validator_kind sid_get_bounds_validator_kind(testee);
            };

            static_assert(sid::concept_impl_::is_sid<testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptr_diff>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::reference_type, testee), element &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::element_type, testee), element>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::const_reference_type, testee), element const &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides_kind>(), "");

            static_assert(std::is_same<decay_t<decltype(sid::get_origin(std::declval<testee &>())())>, ptr>::value, "");
            static_assert(std::is_same<decltype(sid::get_strides(testee{})), strides>(), "");

            static_assert(std::is_same<decay_t<decltype(sid::get_stride<dim_0>(strides{}))>, stride>(), "");
            static_assert(std::is_same<decay_t<decltype(sid::get_stride<dim_1>(strides{}))>, stride>(), "");
            static_assert(sid::get_stride<void>(strides{}) == 0, "");
            static_assert(sid::get_stride<void *>(strides{}) == 0, "");

            static_assert(std::is_same<decltype(sid::shift(std::declval<ptr &>(), stride{}, 0)), std::true_type>(), "");
            static_assert(
                std::is_same<decltype(sid::shift(std::declval<ptr_diff &>(), stride{}, 0)), std::false_type>(), "");
        } // namespace custom

        namespace fallbacks {

            struct testee {
                friend host_device::constant<testee *> sid_get_origin(testee &obj) { return {&obj}; }
            };

            static_assert(is_sid<testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee), testee *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptrdiff_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::reference_type, testee), testee &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::element_type, testee), testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::const_reference_type, testee), testee const &>(), "");

            using strides = GT_META_CALL(sid::strides_type, testee);
            static_assert(tuple_util::size<strides>() == 0, "");

            static_assert(std::is_same<decay_t<decltype(sid::get_origin(std::declval<testee &>())())>, testee *>(), "");
            static_assert(std::is_same<decltype(sid::get_strides(testee{})), strides>(), "");

            constexpr auto stride = sid::get_stride<void>(strides{});
            static_assert(stride == 0, "");

            static_assert(std::is_void<void_t<decltype(sid::shift(std::declval<testee *&>(), stride, 42))>>(), "");
            static_assert(std::is_void<void_t<decltype(sid::shift(std::declval<ptrdiff_t *&>(), stride, 42))>>(), "");
        } // namespace fallbacks

        template <class T, class Stride, class Offset>
        void do_verify_shift(T obj, Stride stride, Offset offset) {
            auto expected = obj + stride * offset;
            sid::shift(obj, stride, offset);
            EXPECT_EQ(expected, obj);
        }

        struct verify_shift_f {
            template <class Stride, class Offset>
            void operator()(Stride stride, Offset offset) const {
                int const data[100] = {};
                do_verify_shift(data + 50, stride, offset);
                do_verify_shift(42, stride, offset);
            }
        };

        TEST(shift, default_overloads) {
            namespace tu = tuple_util;
            auto samples = tu::host_device::make<tuple>(2, 3, -2_c, -1_c, 0_c, 1_c, 2_c);
            tu::host::for_each_in_cartesian_product(verify_shift_f{}, samples, samples);
        }

        namespace non_static_value {
            struct stride {
                int value;
            };

            struct testee {};

            host_device::constant<testee *> sid_get_origin(testee &obj) { return {&obj}; }
            tuple<stride> sid_get_strides(testee const &) { return {}; }
            GT_FUNCTION int operator*(stride, int) { return 100; }
            integral_constant<int, 42> sid_get_strides_kind(testee const &);

            static_assert(is_sid<testee>(), "");

            TEST(non_static_value, shift) {
                ptrdiff_t val = 22;
                sid::shift(val, stride{}, 0);
                EXPECT_EQ(122, val);
                val = 22;
                sid::shift(val, stride{}, 3_c);
                EXPECT_EQ(122, val);
            }
        } // namespace non_static_value

        TEST(c_array, smoke) {
            double testee[15][43] = {};
            static_assert(sid::concept_impl_::is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&testee[0][0], sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 0>>(strides) == 43));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 1>>(strides) == 1));

            using strides_t = decltype(strides);

            static_assert(tuple_util::size<strides_t>::value == 2, "");

            using stride_0_t = GT_META_CALL(tuple_util::element, (0, strides_t));
            using stride_1_t = GT_META_CALL(tuple_util::element, (1, strides_t));

            static_assert(stride_0_t::value == 43, "");
            static_assert(stride_1_t::value == 1, "");

            testee[7][8] = 555;

            auto *ptr = sid::get_origin(testee)();
            sid::shift(ptr, sid::get_stride<integral_constant<int, 0>>(strides), 7);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 1>>(strides), 8);

            EXPECT_EQ(555, *ptr);
        }

        TEST(c_array, 4D) {
            double testee[2][3][4][5] = {};
            static_assert(sid::concept_impl_::is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&testee[0][0][0][0], sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 0>>(strides) == 60));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 1>>(strides) == 20));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 2>>(strides) == 5));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 3>>(strides) == 1));

            testee[1][2][3][4] = 555;

            auto *ptr = sid::get_origin(testee)();
            sid::shift(ptr, sid::get_stride<integral_constant<int, 0>>(strides), 1);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 1>>(strides), 2);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 2>>(strides), 3);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 3>>(strides), 4);

            EXPECT_EQ(555, *ptr);
        }
    } // namespace
} // namespace gridtools
