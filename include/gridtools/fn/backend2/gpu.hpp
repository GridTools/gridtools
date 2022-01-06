/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/cuda_util.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/multi_shift.hpp"

namespace gridtools::fn::backend {
    namespace gpu_impl_ {
        template <class BlockSizes>
        struct gpu {
            using block_sizes_t = BlockSizes;
        };

        template <class BlockSizes, class Dim>
        using block_size_at_dim = meta::second<meta::mp_find<BlockSizes, Dim>>;

        template <class BlockSizes, class Sizes>
        GT_FUNCTION_DEVICE auto global_thread_index() {
            using keys_t = get_keys<Sizes>;
            using ndims_t = meta::length<keys_t>;
            static_assert(ndims_t::value > 0);
            if constexpr (ndims_t::value == 1) {
                using block_dim_x = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 0>>;
                return keys_t::template values<int>(blockIdx.x * block_dim_x::value + threadIdx.x);
            } else if constexpr (ndims_t::value == 2) {
                using block_dim_x = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 0>>;
                using block_dim_y = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 1>>;
                return keys_t::template values<int, int>(
                    blockIdx.x * block_dim_x::value + threadIdx.x, blockIdx.y * block_dim_y::value + threadIdx.y);
            } else {
                using block_dim_x = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 0>>;
                using block_dim_y = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 1>>;
                using block_dim_z = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 2>>;
                return keys_t::template values<int, int, int>(blockIdx.x * block_dim_x::value + threadIdx.x,
                    blockIdx.y * block_dim_y::value + threadIdx.y,
                    blockIdx.z * block_dim_z::value + threadIdx.z);
            }
        }

        template <class Index, class Sizes>
        GT_FUNCTION_DEVICE bool in_domain(Index const &index, Sizes const &sizes) {
            auto in_bounds = tuple_util::device::transform(std::less(), index, sizes);
            return tuple_util::device::fold(std::logical_and(), in_bounds);
        }

        template <class BlockSizes, class Sizes, class PtrHolder, class Strides, class Fun>
        __global__ void kernel(Sizes sizes, PtrHolder ptr_holder, Strides strides, Fun fun) {
            auto thread_idx = global_thread_index<BlockSizes, Sizes>();
            if (!in_domain(thread_idx, sizes))
                return;
            static_assert(meta::length<Sizes>::value <= 3, "higher dimensional computations not implemented");
            auto ptr = ptr_holder();
            sid::multi_shift(ptr, strides, thread_idx);
            fun(ptr, strides);
        }

        template <class BlockSizes, class Sizes>
        std::tuple<dim3, dim3> blocks_and_threads(Sizes const &sizes) {
            using keys_t = get_keys<Sizes>;
            using ndims_t = meta::length<keys_t>;
            dim3 blocks(1, 1, 1);
            dim3 threads(1, 1, 1);
            threads.x = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 0>>::value;
            blocks.x = (tuple_util::get<0>(sizes) + threads.x - 1) / threads.x;
            if constexpr (ndims_t::value >= 2) {
                threads.y = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 1>>::value;
                blocks.y = (tuple_util::get<1>(sizes) + threads.y - 1) / threads.y;
            }
            if constexpr (ndims_t::value >= 3) {
                threads.z = block_size_at_dim<BlockSizes, meta::at_c<keys_t, 2>>::value;
                blocks.z = (tuple_util::get<2>(sizes) + threads.z - 1) / threads.z;
            }
            return {blocks, threads};
        }

        template <class BlockSizes, class Sizes, class StencilStage, class Composite>
        void apply_stencil_stage(gpu<BlockSizes>, Sizes const &sizes, StencilStage, Composite composite) {
            auto ptr_holder = sid::get_origin(composite);
            auto strides = sid::get_strides(composite);

            auto [blocks, threads] = blocks_and_threads<BlockSizes>(sizes);
            cuda_util::launch(blocks,
                threads,
                0,
                kernel<BlockSizes, decltype(ptr_holder), decltype(strides), StencilStage>,
                sizes,
                ptr_holder,
                strides,
                StencilStage());
        }

        template <class ColumnStage, class Seed>
        struct column_fun_f {
            std::size_t m_v_size;
            Seed m_seed;

            template <class Ptr, class Strides>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides) const {
                ColumnStage()(wstd::move(m_seed), m_v_size, wstd::move(ptr), strides);
            }
        };

        template <class Vertical, class BlockSizes, class Sizes, class ColumnStage, class Composite, class Seed>
        void apply_column_stage(gpu<BlockSizes>, Sizes const &sizes, ColumnStage, Composite composite, Seed seed) {
            auto ptr_holder = sid::get_origin(composite);
            auto strides = sid::get_strides(composite);
            auto h_sizes = hymap::remove_key<Vertical>(sizes);
            auto v_size = at_key<Vertical>(sizes);

            auto [blocks, threads] = blocks_and_threads<BlockSizes>(h_sizes);
            cuda_util::launch(blocks,
                threads,
                0,
                kernel<BlockSizes,
                    decltype(h_sizes),
                    decltype(ptr_holder),
                    decltype(strides),
                    column_fun_f<ColumnStage, Seed>>,
                h_sizes,
                ptr_holder,
                strides,
                column_fun_f<ColumnStage, Seed>{v_size, std::move(seed)});
        }
    } // namespace gpu_impl_

    using gpu_impl_::apply_column_stage;
    using gpu_impl_::apply_stencil_stage;
    using gpu_impl_::gpu;
} // namespace gridtools::fn::backend
