/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <gridtools/stencil-composition/sid/delegate.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil-composition/sid/concept.hpp>
#include <gridtools/stencil-composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        template <class Sid>
        class i_shifted : public sid::delegate<Sid> {
            friend GT_META_CALL(sid::ptr_type, Sid) sid_get_origin(i_shifted &obj) {
                auto &&impl = obj.impl();
                auto res = sid::get_origin(impl);
                sid::shift(res, sid::get_stride<1>(sid::get_strides(impl)), 1_c);
                return res;
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
            auto src =
                sid::synthetic().set<property::origin>(&data[0][0]).set<property::strides>(tu::make<tuple>(1_c, 5_c));
            auto testee = i_shift(src);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[0][0], sid::get_origin(src));
            EXPECT_EQ(&data[1][0], sid::get_origin(testee));
        }
    } // namespace
} // namespace gridtools
