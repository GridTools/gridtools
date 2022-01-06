/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/backend2/gpu.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/scan.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn::backend {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return tuple(get<0>(acc) + *iter, get<1>(acc) * *iter); },
                    [](auto acc) { return get<0>(acc); });
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(backend_gpu, apply_column_stage) {
            auto in = cuda_util::cuda_malloc<int>(5 * 7 * 3);
            auto out = cuda_util::cuda_malloc<int>(5 * 7 * 3);
            int inh[5][7][3], outh[5][7][3] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 7; ++j)
                    for (int k = 0; k < 3; ++k)
                        inh[i][j][k] = 21 * i + 3 * j + k;
            cudaMemcpy(in.get(), inh, 5 * 7 * 3 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::make_simple_ptr_holder(x))
                    .set<property::strides>(tuple(21_c, 3_c, 1_c));
            };

            auto composite =
                sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out.get()), as_synthetic(in.get()));

            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>>::values<int_t<5>, int_t<7>, int_t<3>>();

            column_stage<int_t<1>, sum_scan, make_iterator_mock, 0, 1> cs;

            using block_sizes_t = meta::list<meta::list<int_t<0>, int_t<4>>, meta::list<int_t<2>, int_t<2>>>;

            apply_column_stage<int_t<1>>(gpu<block_sizes_t>(), sizes, cs, composite, tuple(42, 1));

            cudaMemcpy(outh, out.get(), 5 * 7 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 3; ++k) {
                    int res = 42;
                    for (int j = 0; j < 7; ++j) {
                        res += inh[i][j][k];
                        EXPECT_EQ(outh[i][j][k], res);
                    }
                }
        }
    } // namespace
} // namespace gridtools::fn::backend
