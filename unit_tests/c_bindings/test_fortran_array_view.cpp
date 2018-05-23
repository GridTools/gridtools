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
namespace {
    bool operator==(gt_fortran_array_descriptor d1, gt_fortran_array_descriptor d2) {
        return d1.type == d2.type && d1.rank == d2.rank &&
               std::equal(std::begin(d1.dims), &d1.dims[d1.rank], std::begin(d2.dims)) && d1.data == d2.data;
    }
}

namespace gridtools {
    namespace adltest {
        namespace {
            template < size_t Rank >
            struct StaticHypercube {
                StaticHypercube(double *data) : data_(data) {}

                double operator()(const std::array< size_t, Rank > &element) {
                    size_t index = 0;
                    for (size_t i = 0; i < Rank; ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index = (element[i] << (Rank - i));
                    }
                    return data_[index];
                }

              private:
                double *data_;
            };

            template < size_t Dimension >
            StaticHypercube< Dimension > gt_make_fortran_array_view(
                gt_fortran_array_descriptor *descriptor, StaticHypercube< Dimension > *) {
                if (descriptor->type != gt_fk_Double) {
                    throw std::runtime_error("type does not match");
                }
                for (int i = 0; i < descriptor->rank; ++i)
                    if (descriptor->dims[i] != 2)
                        throw std::runtime_error("dimensions must be 2");

                return StaticHypercube< Dimension >{reinterpret_cast< double * >(descriptor->data)};
            }
            template < size_t Rank >
            gt_fortran_array_descriptor get_fortran_view_meta(StaticHypercube< Rank > *) {
                return gt_fortran_array_descriptor{gt_fk_Double, Rank};
            }

            struct DynamicHypercube {
                DynamicHypercube(double *data, size_t rank) : data_(data), rank_(rank) {}

                double operator()(const std::vector< size_t > &element) {
                    if (element.size() != rank_) {
                        throw std::out_of_range("Rank out of range");
                    }
                    size_t index = 0;
                    for (size_t i = 0; i < rank_; ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index = (element[i] << (rank_ - i));
                    }
                    return data_[index];
                }

              private:
                double *data_;
                size_t rank_;
            };

            DynamicHypercube gt_make_fortran_array_view(gt_fortran_array_descriptor *descriptor, DynamicHypercube *) {
                if (descriptor->type != gt_fk_Double) {
                    throw std::runtime_error("type does not match");
                }
                for (int i = 0; i < descriptor->rank; ++i)
                    if (descriptor->dims[i] != 2)
                        throw std::runtime_error("dimensions must be 2");

                return DynamicHypercube{reinterpret_cast< double * >(descriptor->data), size_t(descriptor->rank)};
            }
        }
    }
    namespace c_bindings {
        namespace {
            static_assert(is_fortran_array_bindable< gt_fortran_array_descriptor >::value, "");
            static_assert(is_fortran_array_bindable< gt_fortran_array_descriptor & >::value, "");
            static_assert(!is_fortran_array_wrappable< gt_fortran_array_descriptor >::value, "");
            static_assert(!is_fortran_array_wrappable< gt_fortran_array_descriptor & >::value, "");
            TEST(FortranArrayView, FortranArrayDescriptorIsBindable) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{gt_fk_Float, 4, {4, 3, 2, 1}, &data[0]};

                auto new_descriptor = make_fortran_array_view< gt_fortran_array_descriptor >(&descriptor);
            }

            static_assert(is_fortran_array_bindable< int(&)[1][2][3] >::value, "");
            static_assert(!is_fortran_array_bindable< int[1][2][3] >::value, "");
            static_assert(!is_fortran_array_bindable< int (*)[2][3] >::value, "");
            static_assert(is_fortran_array_wrappable< int(&)[1][2][3] >::value, "");
            static_assert(!is_fortran_array_wrappable< int[1][2][3] >::value, "");
            static_assert(!is_fortran_array_wrappable< int (*)[2][3] >::value, "");
            TEST(FortranArrayView, CArrayReferenceIsBindable) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{gt_fk_Float, 4, {4, 3, 2, 1}, &data[0]};

                auto &new_descriptor = make_fortran_array_view< float(&)[1][2][3][4] >(&descriptor);
                static_assert(std::is_same< decltype(new_descriptor), float(&)[1][2][3][4] >::value, "");

                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3][3] >(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[2][2][3][4] >(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3] >(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view< float(&)[1][2][3][4][5] >(&descriptor), std::runtime_error);
            }
            TEST(FortranArrayView, CArrayReferenceIsWrappable) {
                float data[1][2][3][4];
                auto meta = get_fortran_view_meta(decltype(&data)(nullptr));

                ASSERT_TRUE(meta.type == gt_fk_Float);
                ASSERT_TRUE(meta.rank == 4);
            }

            struct NotBindableNotWrappableClass {};
            static_assert(!is_fortran_array_bindable< NotBindableNotWrappableClass >::value, "");
            static_assert(!is_fortran_array_bindable< NotBindableNotWrappableClass & >::value, "");
            static_assert(!is_fortran_array_wrappable< NotBindableNotWrappableClass >::value, "");
            static_assert(!is_fortran_array_wrappable< NotBindableNotWrappableClass & >::value, "");

            struct BindableClassWithConstructor {
                BindableClassWithConstructor(const gt_fortran_array_descriptor &);
            };
            static_assert(is_fortran_array_bindable< BindableClassWithConstructor >::value, "");
            static_assert(!is_fortran_array_bindable< BindableClassWithConstructor & >::value, "");
            static_assert(!is_fortran_array_wrappable< BindableClassWithConstructor >::value, "");
            static_assert(!is_fortran_array_wrappable< BindableClassWithConstructor & >::value, "");

            struct BindableClassWithFactoryFunction {};
            BindableClassWithFactoryFunction gt_make_fortran_array_view(
                gt_fortran_array_descriptor *, BindableClassWithFactoryFunction *) {
                return {};
            }
            static_assert(is_fortran_array_bindable< BindableClassWithFactoryFunction >::value, "");
            static_assert(!is_fortran_array_bindable< BindableClassWithFactoryFunction & >::value, "");
            static_assert(!is_fortran_array_wrappable< BindableClassWithFactoryFunction >::value, "");
            static_assert(!is_fortran_array_wrappable< BindableClassWithFactoryFunction & >::value, "");

            struct WrappableClassWithMetaTypes {
                WrappableClassWithMetaTypes(const gt_fortran_array_descriptor &);
                using gt_view_element_type = double;
                using gt_view_rank = std::integral_constant< size_t, 3 >;
            };
            static_assert(is_fortran_array_bindable< WrappableClassWithMetaTypes >::value, "");
            static_assert(!is_fortran_array_bindable< WrappableClassWithMetaTypes & >::value, "");
            static_assert(is_fortran_array_wrappable< WrappableClassWithMetaTypes >::value, "");
            static_assert(!is_fortran_array_wrappable< WrappableClassWithMetaTypes & >::value, "");

            struct WrappableClassWithMetaFunction {
                WrappableClassWithMetaFunction(const gt_fortran_array_descriptor &);
            };
            gt_fortran_array_descriptor get_fortran_view_meta(WrappableClassWithMetaFunction *) { return {}; };
            static_assert(is_fortran_array_bindable< WrappableClassWithMetaFunction >::value, "");
            static_assert(!is_fortran_array_bindable< WrappableClassWithMetaFunction & >::value, "");
            static_assert(is_fortran_array_wrappable< WrappableClassWithMetaFunction >::value, "");
            static_assert(!is_fortran_array_wrappable< WrappableClassWithMetaFunction & >::value, "");

            static_assert(!is_fortran_array_bindable< adltest::StaticHypercube< 3 > & >::value, "");
            static_assert(is_fortran_array_bindable< adltest::StaticHypercube< 3 > >::value, "");
            static_assert(!is_fortran_array_wrappable< adltest::StaticHypercube< 3 > & >::value, "");
            static_assert(is_fortran_array_wrappable< adltest::StaticHypercube< 3 > >::value, "");
            TEST(FortranArrayView, WrappableArrayIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                auto array = make_fortran_array_view< adltest::StaticHypercube< 4 > >(&descriptor);
                ASSERT_TRUE(data[0][1][0][1] == 6.);
                ASSERT_TRUE(data[1][0][1][0] == 11.);
            }
            TEST(FortranArrayView, WrappableArrayIsWrappable) {
                auto meta = get_fortran_view_meta((adltest::StaticHypercube< 3 > *){nullptr});
                ASSERT_TRUE(meta.type == gt_fk_Double);
                ASSERT_TRUE(meta.rank == 3);
            }

            static_assert(!is_fortran_array_bindable< adltest::DynamicHypercube & >::value, "");
            static_assert(is_fortran_array_bindable< adltest::DynamicHypercube >::value, "");
            static_assert(!is_fortran_array_wrappable< adltest::DynamicHypercube & >::value, "");
            static_assert(!is_fortran_array_wrappable< adltest::DynamicHypercube >::value, "");
            TEST(FortranArrayView, BindableArrayIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                auto array = make_fortran_array_view< adltest::DynamicHypercube >(&descriptor);
                ASSERT_TRUE(data[0][1][0][1] == 6.);
                ASSERT_TRUE(data[1][0][1][0] == 11.);
            }
        }
    }
}
