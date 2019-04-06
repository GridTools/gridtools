/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {
        struct copy_functor {
            GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in), GT_INOUT_ACCESSOR(out));

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = eval(in());
            }
        };

        struct stencils : computation_fixture<> {
            static constexpr uint_t i_max = 58;
            static constexpr uint_t j_max = 46;
            static constexpr uint_t k_max = 70;
            stencils() : computation_fixture<>(i_max + 1, j_max + 1, k_max + 1) {}

            template <class SrcLayout, class DstLayout = layout_map<0, 1, 2>, class Expected>
            void do_test(Expected const &expected) {
                using meta_dst_t = typename storage_tr::
                    select_custom_layout_storage_info<0, DstLayout, zero_halo<DstLayout::masked_length>>::type;
                using meta_src_t = typename storage_tr::
                    select_custom_layout_storage_info<0, SrcLayout, zero_halo<SrcLayout::masked_length>>::type;

                using dst_storage_t = storage_tr::data_store_t<float_type, meta_dst_t>;
                using src_storage_t = storage_tr::data_store_t<float_type, meta_src_t>;

                arg<0, src_storage_t> p_in;
                arg<1, dst_storage_t> p_out;

                auto in = [](int i, int j, int k) { return i + j + k; };
                auto out = make_storage<dst_storage_t>();
                make_computation(p_in = make_storage<src_storage_t>(in),
                    p_out = out,
                    make_multistage(execute::forward(), make_stage<copy_functor>(p_in, p_out)))
                    .run();
                verify(make_storage<dst_storage_t>(expected), out);
            }
        };

        TEST_F(stencils, copies3D) {
            do_test<layout_map<0, 1, 2>>([](int i, int j, int k) { return i + j + k; });
        }

        TEST_F(stencils, copies3Dtranspose) {
            do_test<layout_map<2, 1, 0>>([](int i, int j, int k) { return i + j + k; });
        }

        TEST_F(stencils, copies2Dij) {
            do_test<layout_map<0, 1, -1>>([](int i, int j, int) { return i + j + k_max; });
        }

        TEST_F(stencils, copies2Dik) {
            do_test<layout_map<0, -1, 1>>([](int i, int, int k) { return i + j_max + k; });
        }

        TEST_F(stencils, copies2Djk) {
            do_test<layout_map<-1, 0, 1>>([](int, int j, int k) { return i_max + j + k; });
        }

        TEST_F(stencils, copies1Di) {
            do_test<layout_map<0, -1, -1>>([](int i, int, int) { return i + j_max + k_max; });
        }

        TEST_F(stencils, copies1Dj) {
            do_test<layout_map<-1, 0, -1>>([](int, int j, int) { return i_max + j + k_max; });
        }

        TEST_F(stencils, copies1Dk) {
            do_test<layout_map<-1, -1, 0>>([](int, int, int k) { return i_max + j_max + k; });
        }

        TEST_F(stencils, copiesScalar) {
            do_test<layout_map<-1, -1, -1>>([](int, int, int) { return i_max + j_max + k_max; });
        }

        TEST_F(stencils, copies3DDst) {
            do_test<layout_map<0, 1, 2>, layout_map<2, 0, 1>>([](int i, int j, int k) { return i + j + k; });
        }

        TEST_F(stencils, copies3DtransposeDst) {
            do_test<layout_map<2, 1, 0>, layout_map<2, 0, 1>>([](int i, int j, int k) { return i + j + k; });
        }

        TEST_F(stencils, copies2DijDst) {
            do_test<layout_map<1, 0, -1>, layout_map<2, 0, 1>>([](int i, int j, int) { return i + j + k_max; });
        }

        TEST_F(stencils, copies2DikDst) {
            do_test<layout_map<1, -1, 0>, layout_map<2, 0, 1>>([](int i, int, int k) { return i + j_max + k; });
        }

        TEST_F(stencils, copies2DjkDst) {
            do_test<layout_map<-1, 1, 0>, layout_map<2, 0, 1>>([](int, int j, int k) { return i_max + j + k; });
        }

        TEST_F(stencils, copies2DiDst) {
            do_test<layout_map<0, -1, -1>, layout_map<2, 0, 1>>([](int i, int, int) { return i + j_max + k_max; });
        }

        TEST_F(stencils, copies2DjDst) {
            do_test<layout_map<-1, 0, -1>, layout_map<2, 0, 1>>([](int, int j, int) { return i_max + j + k_max; });
        }

        TEST_F(stencils, copies2DkDst) {
            do_test<layout_map<-1, -1, 0>, layout_map<2, 0, 1>>([](int, int, int k) { return i_max + j_max + k; });
        }

        TEST_F(stencils, copies2DScalarDst) {
            do_test<layout_map<-1, -1, -1>, layout_map<2, 0, 1>>([](int, int, int) { return i_max + j_max + k_max; });
        }
    } // namespace
} // namespace gridtools
