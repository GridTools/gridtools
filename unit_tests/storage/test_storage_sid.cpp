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

#include <gridtools/storage/sid.hpp>

#include <type_traits>

#include <gtest/gtest.h>

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

            EXPECT_EQ(get<0>(strides), 10);
            EXPECT_EQ(get<1>(strides), 0);
            EXPECT_EQ(get<2>(strides), 1);
            EXPECT_EQ(get<3>(strides), 100);
        }

        TEST(storage_sid, const_smoke) {
            data_store_t data = {{10, 10, 10, 10}, 0};
            auto testee = as_const(data);

            using testee_t = decltype(testee);

            static_assert(is_sid<testee_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee_t), float_type const *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee_t),
                              GT_META_CALL(sid::ptr_diff_type, data_store_t)>(),
                "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee_t),
                              GT_META_CALL(sid::strides_kind, data_store_t)>(),
                "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_type, testee_t),
                              GT_META_CALL(sid::strides_type, data_store_t)>(),
                "");

            EXPECT_EQ(sid::get_origin(data), sid::get_origin(testee));

            auto testee_strides = sid::get_strides(testee);
            auto data_strides = sid::get_strides(data);

            EXPECT_EQ(get<0>(data_strides), get<0>(testee_strides));
            EXPECT_EQ(get<1>(data_strides), get<1>(testee_strides));
            EXPECT_EQ(get<2>(data_strides), get<2>(testee_strides));
            EXPECT_EQ(get<3>(data_strides), get<3>(testee_strides));
        }
    } // namespace
} // namespace gridtools
