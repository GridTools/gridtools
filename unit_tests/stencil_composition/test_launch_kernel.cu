/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/launch_kernel.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/extent.hpp>

namespace gridtools {
    namespace cuda {
        template <class Extent, int_t IBlockSize, int_t JBlockSize>
        struct kernel_f {
            int *m_failures;
            int_t m_i_size;
            int_t m_j_size;

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(int_t iblock, int_t jblock, Validator is_valid) const {
                int_t i_block_size =
                    (blockIdx.x + 1) * IBlockSize < m_i_size ? IBlockSize : m_i_size - blockIdx.x * IBlockSize;
                int_t j_block_size =
                    (blockIdx.y + 1) * JBlockSize < m_j_size ? JBlockSize : m_j_size - blockIdx.y * JBlockSize;
                bool expected = Extent::iminus::value <= iblock && Extent::iplus::value + i_block_size > iblock &&
                                Extent::jminus::value <= jblock && Extent::jplus::value + j_block_size > jblock;
                bool actual = is_valid(Extent());
                if (actual == expected)
                    return;
                atomicAdd(m_failures, 1);
                printf("failure at {%d,%d} of block {%d,%d}: false %s\n",
                    iblock,
                    jblock,
                    blockIdx.x,
                    blockIdx.y,
                    actual ? "positive" : "negative");
            }
        };

        template <class MaxExtent, class Extent, int_t IBlockSize, int_t JBlockSize>
        void do_test(int_t i_size, int_t j_size) {
            auto failures = cuda_util::make_clone(0);
            kernel_f<Extent, IBlockSize, JBlockSize> kernel = {failures.get(), i_size, j_size};
            launch_kernel<MaxExtent, IBlockSize, JBlockSize>(i_size, j_size, 1, kernel, 0);
            EXPECT_EQ(0, cuda_util::from_clone(failures));
        }

        TEST(launch_kernel, simplest) { do_test<extent<>, extent<>, 32, 8>(128, 128); }

        TEST(launch_kernel, rounded_sizes) { do_test<extent<-2, 2, -1, 3>, extent<-1, 1, 0, 2>, 32, 8>(128, 128); }

        TEST(launch_kernel, hori_diff) { do_test<extent<-1, 1, -1, 1>, extent<-1, 1, -1, 1>, 32, 8>(128, 128); }

        TEST(launch_kernel, hori_diff_small_size) { do_test<extent<-1, 1, -1, 1>, extent<-1, 1, -1, 1>, 32, 8>(5, 5); }

        TEST(launch_kernel, max_extent) { do_test<extent<-2, 2, -1, 3>, extent<-2, 2, -1, 3>, 32, 8>(123, 50); }

        TEST(launch_kernel, zero_extent) { do_test<extent<-2, 2, -1, 3>, extent<>, 32, 8>(123, 50); }

        TEST(launch_kernel, reduced_extent) { do_test<extent<-2, 2, -1, 3>, extent<-1, 1, 0, 2>, 32, 8>(123, 50); }
    } // namespace cuda
} // namespace gridtools
