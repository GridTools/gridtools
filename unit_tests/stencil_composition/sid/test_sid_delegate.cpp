/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/sid/delegate.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/functional.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/stencil_composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        template <class Sid>
        class i_shifted : public sid::delegate<Sid> {
            friend host_device::constant<GT_META_CALL(sid::ptr_type, Sid)> sid_get_origin(i_shifted &obj) {
                auto &&impl = obj.impl();
                auto res = sid::get_origin(impl)();
                sid::shift(res, sid::get_stride<integral_constant<int, 1>>(sid::get_strides(impl)), 1_c);
                return {res};
            }
            using sid::delegate<Sid>::delegate;
        };

        template <class Sid>
        i_shifted<Sid> i_shift(Sid const &sid) {
            return i_shifted<Sid>{sid};
        }

        using sid::property;
        namespace tu = tuple_util;

        TEST(delegate, smoke) {
            double data[3][5];
            auto src = sid::synthetic()
                           .set<property::origin>(host_device::constant<double *>{&data[0][0]})
                           .set<property::strides>(tu::make<tuple>(1_c, 5_c));
            auto testee = i_shift(src);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[0][0], sid::get_origin(src)());
            EXPECT_EQ(&data[1][0], sid::get_origin(testee)());
        }
    } // namespace
} // namespace gridtools
