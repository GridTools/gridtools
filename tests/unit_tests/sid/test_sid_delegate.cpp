/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/delegate.hpp>

#include <functional>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/simple_ptr_holder.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        template <class Sid>
        class i_shifted : public sid::delegate<Sid> {
            friend sid::ptr_holder_type<Sid> sid_get_origin(i_shifted &obj) {
                auto &&impl = obj.impl();
                sid::ptr_diff_type<Sid> offset{};
                sid::shift(offset, sid::get_stride<integral_constant<int, 0>>(sid::get_strides(impl)), 1_c);
                return sid::get_origin(impl) + offset;
            }

          public:
            template <class Arg>
            i_shifted(Arg &&arg) : sid::delegate<Sid>(std::forward<Arg>(arg)) {}
        };

        template <class Sid>
        i_shifted<Sid> i_shift(Sid const &sid) {
            return {sid};
        }

        template <class Sid>
        i_shifted<Sid &> i_shift(std::reference_wrapper<Sid> sid) {
            return {sid.get()};
        }

        using sid::property;
        namespace tu = tuple_util;

        TEST(delegate, smoke) {
            double data[3][5];
            auto src = sid::synthetic()
                           .set<property::origin>(sid::host_device::make_simple_ptr_holder(&data[0][0]))
                           .set<property::strides>(tu::make<tuple>(5_c, 1_c));
            auto testee = i_shift(src);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[0][0], sid::get_origin(src)());
            EXPECT_EQ(&data[1][0], sid::get_origin(testee)());
        }

        TEST(delegate, array) {
            double src[3][5];
            auto testee = i_shift(std::ref(src));

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&src[1][0], sid::get_origin(testee)());
        }

        template <class Sid>
        struct delegate_everything : sid::delegate<Sid> {
            using sid::delegate<Sid>::delegate;
        };

        template <class Sid>
        delegate_everything<Sid> just_delegate(Sid const &s) {
            return delegate_everything<Sid>(s);
        }

        TEST(delegate, do_nothing) {
            double data[3][5];
            auto src = sid::synthetic()
                           .set<property::origin>(sid::host_device::make_simple_ptr_holder(&data[0][0]))
                           .set<property::strides>(tu::make<tuple>(5_c, 1_c));
            auto testee = just_delegate(src);
            static_assert(is_sid<decltype(testee)>(), "");
        }

    } // namespace
} // namespace gridtools
