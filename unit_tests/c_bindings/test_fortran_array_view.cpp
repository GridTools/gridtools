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

#include <c_bindings/fortran_array_view.hpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace other {
        namespace {
            struct X {
                gt_fortran_array_descriptor data;
            };

            X gt_make_fortran_array_view(gt_fortran_array_descriptor *descriptor, X *) { return X{*descriptor}; }
        }
    }
    namespace c_bindings {
        namespace {
            bool IsSameArrayDescriptor(gt_fortran_array_descriptor *d1, gt_fortran_array_descriptor *d2) {
                return d1->type == d2->type && d1->rank == d2->rank &&
                       std::equal(std::begin(d1->dims), std::end(d1->dims), std::begin(d2->dims)) &&
                       d1->data == d2->data;
            }

            // TODO As long the "inspectable"-part is static, gt_fortran_array_descriptor is not viewable
            static_assert(false == is_fortran_array_convertible< gt_fortran_array_descriptor >::value, "");
            static_assert(false == is_fortran_array_convertible< gt_fortran_array_descriptor & >::value, "");

            static_assert(false == is_fortran_array_viewable< gt_fortran_array_descriptor >::value, "");
            static_assert(false == is_fortran_array_viewable< gt_fortran_array_descriptor & >::value, "");

            static_assert(true == is_fortran_array_convertible< int(&)[1][2][3] >::value, "");
            static_assert(false == is_fortran_array_convertible< int[1][2][3] >::value, "");
            static_assert(false == is_fortran_array_convertible< int (*)[2][3] >::value, "");

            static_assert(true == is_fortran_array_viewable< int(&)[1][2][3] >::value, "");
            static_assert(false == is_fortran_array_viewable< int[1][2][3] >::value, "");
            static_assert(false == is_fortran_array_viewable< int (*)[2][3] >::value, "");

            struct C {};
            static_assert(false == is_fortran_array_convertible< C >::value, "");
            static_assert(false == is_fortran_array_convertible< C & >::value, "");

            static_assert(false == is_fortran_array_viewable< C >::value, "");
            static_assert(false == is_fortran_array_viewable< C & >::value, "");

            struct D {
                D(const gt_fortran_array_descriptor &data_) : data(data_) {}
                gt_fortran_array_descriptor data;
            };
            static_assert(false == is_fortran_array_convertible< D & >::value, "");
            static_assert(true == is_fortran_array_convertible< D >::value, "");
            static_assert(false == is_fortran_array_viewable< D & >::value, "");
            static_assert(false == is_fortran_array_viewable< D >::value, "");

            static_assert(false == is_fortran_array_convertible< other::X & >::value, "");
            static_assert(true == is_fortran_array_convertible< other::X >::value, "");
            static_assert(false == is_fortran_array_viewable< other::X & >::value, "");
            static_assert(false == is_fortran_array_viewable< other::X >::value, "");

            struct E {
                E(const gt_fortran_array_descriptor &) {}
            };
        }
        template <>
        struct fortran_array_view_rank< E > : std::integral_constant< std::size_t, 3 > {};
        template <>
        struct fortran_array_view_element_type< E > {
            using type = int;
        };
        namespace {
            static_assert(false == is_fortran_array_convertible< E & >::value, "");
            static_assert(true == is_fortran_array_convertible< E >::value, "");
            static_assert(false == is_fortran_array_viewable< E & >::value, "");
            static_assert(true == is_fortran_array_viewable< E >::value, "");

            struct F {
                F(const gt_fortran_array_descriptor &data_) : data(data_) {}
                gt_fortran_array_descriptor data;

                using gt_view_element_type = int;
                using gt_view_rank = std::integral_constant< std::size_t, 3 >;
            };
            static_assert(true == is_fortran_array_convertible< F >::value, "");
            static_assert(true == is_fortran_array_viewable< F >::value, "");

            TEST(FortranArrayView, ToArray) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{0, 4, {4, 3, 2, 1}, &data[0]};

                auto &new_descriptor = make_fortran_array_view< float(&)[1][2][3][4] >(descriptor);
                static_assert(std::is_same< decltype(new_descriptor), float(&)[1][2][3][4] >::value, "");

                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3][3] >(descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[2][2][3][4] >(descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3] >(descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3][4][5] >(descriptor), std::runtime_error);
            }
            TEST(FortranArrayView, ByConversion) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{0, 4, {4, 3, 2, 1}, &data[0]};

                auto new_descriptor = make_fortran_array_view< D >(descriptor);
                ASSERT_PRED2(IsSameArrayDescriptor, &new_descriptor.data, &descriptor);
            }
            TEST(FortranArrayView, ByFunction) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{0, 4, {4, 3, 2, 1}, &data[0]};

                auto new_descriptor = make_fortran_array_view< other::X >(descriptor);
                ASSERT_PRED2(IsSameArrayDescriptor, &new_descriptor.data, &descriptor);
            }
        }
    }
}
