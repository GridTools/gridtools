/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/c_bindings/fortran_array_view.hpp>

#include <gtest/gtest.h>

bool operator==(const gt_fortran_array_descriptor &d1, const gt_fortran_array_descriptor &d2) {
    return d1.type == d2.type && d1.rank == d2.rank &&
           std::equal(std::begin(d1.dims), &d1.dims[d1.rank], std::begin(d2.dims)) && d1.data == d2.data;
}
bool operator!=(const gt_fortran_array_descriptor &d1, const gt_fortran_array_descriptor &d2) { return !(d1 == d2); }
std::ostream &operator<<(std::ostream &strm, const gt_fortran_array_descriptor &d) {
    strm << "Type: " << d.type << ", Dimensions: [";
    for (size_t i = 0; i < d.rank; ++i) {
        if (i)
            strm << ", ";
        strm << d.dims[i];
    }
    strm << "], Data: " << d.data;
    return strm;
}

namespace gridtools {
    namespace adltest {
        namespace {
            template <size_t Rank>
            struct StaticHypercube {
                StaticHypercube(double *data) : data_(data) {}

                template <class... Ts>
                double operator()(Ts... indices) const {
                    size_t element[Rank] = {size_t(indices)...};
                    size_t index = 0;
                    for (size_t i = 0; i < Rank; ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index += (element[i] << (Rank - i - 1));
                    }
                    return data_[index];
                }

              private:
                double *data_;
            };

            template <size_t Rank>
            StaticHypercube<Rank> gt_make_fortran_array_view(
                gt_fortran_array_descriptor *descriptor, StaticHypercube<Rank> *) {
                if (descriptor->type != gt_fk_Double) {
                    throw std::runtime_error("type does not match");
                }
                for (int i = 0; i < descriptor->rank; ++i)
                    if (descriptor->dims[i] != 2)
                        throw std::runtime_error("dimensions must be 2");

                return StaticHypercube<Rank>{reinterpret_cast<double *>(descriptor->data)};
            }
            template <size_t Rank>
            gt_fortran_array_descriptor get_fortran_view_meta(StaticHypercube<Rank> *) {
                gt_fortran_array_descriptor descriptor;
                descriptor.rank = Rank;
                descriptor.type = gt_fk_Double;
                descriptor.is_acc_present = false;
                for (size_t i = 0; i < Rank; ++i) {
                    descriptor.dims[i] = 2;
                }
                return descriptor;
            }

            struct DynamicHypercube {
                DynamicHypercube(double *data, size_t rank) : data_(data), rank_(rank) {}

                template <class... Ts>
                double operator()(Ts... indices) const {
                    size_t element[] = {size_t(indices)...};
                    if (sizeof...(Ts) != rank_) {
                        throw std::out_of_range("Rank out of range");
                    }
                    size_t index = 0;
                    for (size_t i = 0; i < rank_; ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index += (element[i] << (rank_ - i - 1));
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

                return DynamicHypercube{reinterpret_cast<double *>(descriptor->data), size_t(descriptor->rank)};
            }
        } // namespace
    }     // namespace adltest
    namespace c_bindings {
        namespace {
            static_assert(is_fortran_array_bindable<gt_fortran_array_descriptor>::value, "");
            static_assert(is_fortran_array_bindable<gt_fortran_array_descriptor &>::value, "");
            static_assert(!is_fortran_array_wrappable<gt_fortran_array_descriptor>::value, "");
            static_assert(!is_fortran_array_wrappable<gt_fortran_array_descriptor &>::value, "");
            TEST(FortranArrayView, FortranArrayDescriptorIsBindable) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{gt_fk_Float, 4, {4, 3, 2, 1}, &data[0]};

                auto new_descriptor = make_fortran_array_view<gt_fortran_array_descriptor>(&descriptor);
                EXPECT_EQ(new_descriptor, descriptor);
            }

            static_assert(is_fortran_array_bindable<int (&)[1][2][3]>::value, "");
            static_assert(!is_fortran_array_bindable<int[1][2][3]>::value, "");
            static_assert(!is_fortran_array_bindable<int (*)[2][3]>::value, "");
            static_assert(is_fortran_array_wrappable<int (&)[1][2][3]>::value, "");
            static_assert(!is_fortran_array_wrappable<int[1][2][3]>::value, "");
            static_assert(!is_fortran_array_wrappable<int (*)[2][3]>::value, "");
            TEST(FortranArrayView, CArrayReferenceIsBindable) {
                float data[1][2][3][4];
                gt_fortran_array_descriptor descriptor{gt_fk_Float, 4, {4, 3, 2, 1}, &data[0]};

                auto &view = make_fortran_array_view<float(&)[1][2][3][4]>(&descriptor);
                static_assert(std::is_same<decltype(view), float(&)[1][2][3][4]>::value, "");
                EXPECT_EQ(view, descriptor.data);

                EXPECT_THROW(make_fortran_array_view<float(&)[1][2][3][3]>(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view<float(&)[2][2][3][4]>(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view<float(&)[1][2][3]>(&descriptor), std::runtime_error);
                EXPECT_THROW(make_fortran_array_view<float(&)[1][2][3][4][5]>(&descriptor), std::runtime_error);
            }
            TEST(FortranArrayView, CArrayReferenceIsWrappable) {
                float data[1][2][3][4];
                auto meta = get_fortran_view_meta(decltype (&data)(nullptr));

                EXPECT_EQ(meta.type, gt_fk_Float);
                ASSERT_EQ(meta.rank, 4);
                EXPECT_EQ(meta.dims[0], 1);
                EXPECT_EQ(meta.dims[1], 2);
                EXPECT_EQ(meta.dims[2], 3);
                EXPECT_EQ(meta.dims[3], 4);
            }

            struct NotBindableNotWrappableClass {};
            static_assert(!is_fortran_array_bindable<NotBindableNotWrappableClass>::value, "");
            static_assert(!is_fortran_array_bindable<NotBindableNotWrappableClass &>::value, "");
            static_assert(!is_fortran_array_wrappable<NotBindableNotWrappableClass>::value, "");
            static_assert(!is_fortran_array_wrappable<NotBindableNotWrappableClass &>::value, "");

            template <size_t Rank>
            struct BindableStaticHypercubeWithConstructor {
                BindableStaticHypercubeWithConstructor(const gt_fortran_array_descriptor &descriptor) {
                    if (descriptor.type != gt_fk_Double) {
                        throw std::runtime_error("type does not match");
                    }
                    for (int i = 0; i < descriptor.rank; ++i)
                        if (descriptor.dims[i] != 2)
                            throw std::runtime_error("dimensions must be 2");

                    data_ = reinterpret_cast<double *>(descriptor.data);
                }

                template <class... Ts>
                double operator()(Ts... indices) const {
                    size_t element[Rank] = {size_t(indices)...};
                    size_t index = 0;
                    for (size_t i = 0; i < sizeof...(Ts); ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index += (element[i] << (Rank - i - 1));
                    }
                    return data_[index];
                }

              private:
                double *data_;
            };
            static_assert(is_fortran_array_bindable<BindableStaticHypercubeWithConstructor<2>>::value, "");
            static_assert(!is_fortran_array_bindable<BindableStaticHypercubeWithConstructor<2> &>::value, "");
            static_assert(!is_fortran_array_wrappable<BindableStaticHypercubeWithConstructor<2>>::value, "");
            static_assert(!is_fortran_array_wrappable<BindableStaticHypercubeWithConstructor<2> &>::value, "");
            TEST(FortranArrayView, BindableStaticHypercubeWithConstructorIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                BindableStaticHypercubeWithConstructor<4> view =
                    make_fortran_array_view<BindableStaticHypercubeWithConstructor<4>>(&descriptor);
                EXPECT_EQ(view(0, 1, 0, 1), 6.);
                EXPECT_EQ(view(1, 0, 1, 0), 11.);
            }

            template <size_t Rank>
            struct WrappableStaticHypercubeWithMetaTypes {
                WrappableStaticHypercubeWithMetaTypes(const gt_fortran_array_descriptor &descriptor) {
                    if (descriptor.type != gt_fk_Double) {
                        throw std::runtime_error("type does not match");
                    }
                    for (int i = 0; i < descriptor.rank; ++i)
                        if (descriptor.dims[i] != 2)
                            throw std::runtime_error("dimensions must be 2");

                    data_ = reinterpret_cast<double *>(descriptor.data);
                }

                template <class... Ts>
                double operator()(Ts... indices) const {
                    size_t element[Rank] = {size_t(indices)...};
                    size_t index = 0;
                    for (size_t i = 0; i < Rank; ++i) {
                        if (element[i] >= 2) {
                            throw std::out_of_range("Index out of range");
                        }
                        index += (element[i] << (Rank - i - 1));
                    }
                    return data_[index];
                }

                using gt_view_element_type = double;
                using gt_view_rank = std::integral_constant<size_t, Rank>;
                using gt_is_acc_present = bool_constant<false>;

              private:
                double *data_;
            };
            static_assert(is_fortran_array_bindable<WrappableStaticHypercubeWithMetaTypes<3>>::value, "");
            static_assert(!is_fortran_array_bindable<WrappableStaticHypercubeWithMetaTypes<3> &>::value, "");
            static_assert(is_fortran_array_wrappable<WrappableStaticHypercubeWithMetaTypes<3>>::value, "");
            static_assert(!is_fortran_array_wrappable<WrappableStaticHypercubeWithMetaTypes<3> &>::value, "");
            TEST(FortranArrayView, WrappableStaticHypercubeWithMetaTypesIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                WrappableStaticHypercubeWithMetaTypes<4> view =
                    make_fortran_array_view<WrappableStaticHypercubeWithMetaTypes<4>>(&descriptor);
                EXPECT_EQ(view(0, 1, 0, 1), 6.);
                EXPECT_EQ(view(1, 0, 1, 0), 11.);
            }

            TEST(FortranArrayView, WrappableStaticHypercubeWithMetaTypesIsWrappable) {
                auto meta = get_fortran_view_meta((WrappableStaticHypercubeWithMetaTypes<3> *){nullptr});
                EXPECT_EQ(meta.type, gt_fk_Double);
                EXPECT_EQ(meta.rank, 3);
            }

            static_assert(!is_fortran_array_bindable<adltest::DynamicHypercube &>::value, "");
            static_assert(is_fortran_array_bindable<adltest::DynamicHypercube>::value, "");
            static_assert(!is_fortran_array_wrappable<adltest::DynamicHypercube &>::value, "");
            static_assert(!is_fortran_array_wrappable<adltest::DynamicHypercube>::value, "");
            TEST(FortranArrayView, BindableDynamicHypercubeWithFactoryFunctionIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                adltest::DynamicHypercube view = make_fortran_array_view<adltest::DynamicHypercube>(&descriptor);
                EXPECT_EQ(view(0, 1, 0, 1), 6.);
                EXPECT_EQ(view(1, 0, 1, 0), 11.);
            }

            static_assert(!is_fortran_array_bindable<adltest::StaticHypercube<3> &>::value, "");
            static_assert(is_fortran_array_bindable<adltest::StaticHypercube<3>>::value, "");
            static_assert(!is_fortran_array_wrappable<adltest::StaticHypercube<3> &>::value, "");
            static_assert(is_fortran_array_wrappable<adltest::StaticHypercube<3>>::value, "");
            TEST(FortranArrayView, WrappableStaticHypercubeWithMetaFunctionIsBindable) {
                double data[2][2][2][2] = {
                    {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}, {{{9., 10.}, {11., 12.}}, {{13., 14.}, {15., 16.}}}};
                gt_fortran_array_descriptor descriptor{gt_fk_Double, 4, {2, 2, 2, 2}, &data[0]};

                adltest::StaticHypercube<4> view = make_fortran_array_view<adltest::StaticHypercube<4>>(&descriptor);
                EXPECT_EQ(view(0, 1, 0, 1), 6.);
                EXPECT_EQ(view(1, 0, 1, 0), 11.);
            }
            TEST(FortranArrayView, WrappableStaticHypercubeWithMetaFunctionIsWrappable) {
                gt_fortran_array_descriptor meta = get_fortran_view_meta((adltest::StaticHypercube<3> *){nullptr});
                EXPECT_EQ(meta.type, gt_fk_Double);
                EXPECT_EQ(meta.rank, 3);
            }
        } // namespace
    }     // namespace c_bindings
} // namespace gridtools
