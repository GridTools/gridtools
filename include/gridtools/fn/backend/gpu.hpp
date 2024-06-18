/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <utility>

#include "../../common/cuda_util.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/multi_shift.hpp"
#include "../../sid/unknown_kind.hpp"
#include "./common.hpp"

namespace gridtools::fn::backend {
    namespace gpu_impl_ {
        /*
         * ThreadBlockSizes must be a meta map, mapping dimensions to integral constant block sizes.
         *
         * For example, meta::list<meta::list<dim::i, integral_constant<int, 32>>,
         *                         meta::list<dim::j, integral_constant<int, 8>>,
         *                         meta::list<dim::k, integral_constant<int, 1>>>;
         * When using a cartesian grid.
         */
        template <class ThreadBlockSizes,
            class LoopBlockSizes = meta::transpose<meta::list<meta::first<meta::transpose<ThreadBlockSizes>>,
                meta::repeat<meta::length<ThreadBlockSizes>, meta::list<integral_constant<int, 1>>>>>>
        struct gpu {
            using thread_block_sizes_t = ThreadBlockSizes;
            using loop_block_sizes_t = LoopBlockSizes;
            cudaStream_t stream = 0;
        };

        template <class BlockSizes>
        struct block_size_at_dim {
            template <class Dim>
            using apply = std::conditional_t<std::is_void_v<meta::mp_find<BlockSizes, Dim, void>>,
                meta::list<Dim, integral_constant<int, 1>>,
                meta::mp_find<BlockSizes, Dim, void>>;
        };

        template <class BlockSizes, class Sizes>
        using block_sizes_for_sizes =
            hymap::from_meta_map<meta::transform<block_size_at_dim<BlockSizes>::apply, get_keys<Sizes>>>;

        struct extract_uint3_f {
            uint3 const &values;

            template <size_t I, class Value>
            GT_FUNCTION_DEVICE constexpr void operator()(Value &value) const {
                if constexpr (I == 0)
                    value = values.x;
                else if constexpr (I == 1)
                    value = values.y;
                else if constexpr (I == 2)
                    value = values.z;
            }
        };

        struct global_thread_index_f {
            template <class BlockIndex, class ThreadIndex, class ThreadBlockSize, class LoopBlockSize>
            GT_FUNCTION_DEVICE constexpr auto operator()(
                BlockIndex block_index, ThreadIndex thread_index, ThreadBlockSize, LoopBlockSize) const {
                return block_index * (ThreadBlockSize::value * LoopBlockSize::value) +
                       thread_index * LoopBlockSize::value;
            }
        };

        struct block_size_f {
            template <size_t I, class GlobalThreadIndex, class LoopBlockSize, class Size>
            GT_FUNCTION_DEVICE constexpr auto operator()(
                GlobalThreadIndex global_thread_index, LoopBlockSize, Size size) const {
                if constexpr (I < 3) {
                    if constexpr (LoopBlockSize::value == 1)
                        return integral_constant<int, 1>();
                    else
                        return std::clamp(size - global_thread_index, 0, int(LoopBlockSize::value));
                } else {
                    return size;
                }
                // disable incorrect warning "missing return statement at end of non-void function"
                GT_NVCC_DIAG_PUSH_SUPPRESS(940)
            }
            GT_NVCC_DIAG_POP_SUPPRESS(940)
        };

        template <class ThreadBlockSizes, class LoopBlockSizes, class Sizes>
        GT_FUNCTION_DEVICE constexpr auto global_thread_index(Sizes const &sizes) {
            using keys_t = meta::rename<hymap::keys, get_keys<Sizes>>;

            constexpr int dynamic_indices = std::min(int(meta::length<keys_t>::value), 3);
            constexpr int static_indices = meta::length<keys_t>::value - dynamic_indices;
            using indices_t = meta::rename<keys_t::values,
                meta::concat<meta::repeat_c<dynamic_indices, meta::list<int>>,
                    meta::repeat_c<static_indices, meta::list<integral_constant<int, 0>>>>>;
            indices_t thread_indices, block_indices;
            tuple_util::device::for_each_index(extract_uint3_f{threadIdx}, thread_indices);
            tuple_util::device::for_each_index(extract_uint3_f{blockIdx}, block_indices);

            constexpr auto thread_block_sizes = block_sizes_for_sizes<ThreadBlockSizes, Sizes>();
            constexpr auto loop_block_sizes = block_sizes_for_sizes<LoopBlockSizes, Sizes>();

            auto global_thread_indices = tuple_util::device::transform(
                global_thread_index_f{}, block_indices, thread_indices, thread_block_sizes, loop_block_sizes);
            auto block_sizes =
                tuple_util::device::transform_index(block_size_f{}, global_thread_indices, loop_block_sizes, sizes);
            return std::make_tuple(std::move(global_thread_indices), std::move(block_sizes));
        }

        template <class Int, Int... i>
        constexpr int iseq_product(std::integer_sequence<Int, i...>) {
            return (1 * ... * i);
        }

        template <class ThreadBlockSizes,
            class LoopBlockSizes,
            class Sizes,
            class PtrHolder,
            class Strides,
            class Fun,
            int NumThreads = iseq_product(meta::list_to_iseq<block_sizes_for_sizes<ThreadBlockSizes, Sizes>>())>
        __global__ void __launch_bounds__(NumThreads)
            kernel(Sizes sizes, PtrHolder ptr_holder, Strides strides, Fun fun) {
            auto const [thread_idx, block_size] = global_thread_index<ThreadBlockSizes, LoopBlockSizes>(sizes);
            if (!tuple_util::all_of(std::less(), thread_idx, sizes))
                return;
            auto ptr = ptr_holder();
            sid::multi_shift(ptr, strides, thread_idx);
            common::make_loops(block_size)(std::move(fun))(ptr, strides);
        }

        template <class ThreadBlockSizes, class LoopBlockSizes, class Sizes>
        std::tuple<dim3, dim3> blocks_and_threads(Sizes const &sizes) {
            using keys_t = get_keys<Sizes>;
            using ndims_t = meta::length<keys_t>;
            [[maybe_unused]] constexpr auto thread_block_sizes = block_sizes_for_sizes<ThreadBlockSizes, Sizes>();
            [[maybe_unused]] constexpr auto loop_block_sizes = block_sizes_for_sizes<LoopBlockSizes, Sizes>();
            dim3 blocks(1, 1, 1);
            dim3 threads(1, 1, 1);
            if constexpr (ndims_t::value >= 1) {
                threads.x = tuple_util::get<0>(thread_block_sizes);
                constexpr int block_dim_x =
                    tuple_util::get<0>(thread_block_sizes) * tuple_util::get<0>(loop_block_sizes);
                blocks.x = (tuple_util::get<0>(sizes) + block_dim_x - 1) / block_dim_x;
            }
            if constexpr (ndims_t::value >= 2) {
                threads.y = tuple_util::get<1>(thread_block_sizes);
                constexpr int block_dim_y =
                    tuple_util::get<1>(thread_block_sizes) * tuple_util::get<1>(loop_block_sizes);
                blocks.y = (tuple_util::get<1>(sizes) + block_dim_y - 1) / block_dim_y;
            }
            if constexpr (ndims_t::value >= 3) {
                threads.z = tuple_util::get<2>(thread_block_sizes);
                constexpr int block_dim_z =
                    tuple_util::get<2>(thread_block_sizes) * tuple_util::get<2>(loop_block_sizes);
                blocks.z = (tuple_util::get<2>(sizes) + block_dim_z - 1) / block_dim_z;
            }
            return {blocks, threads};
        }

        template <class StencilStage, class MakeIterator>
        struct stencil_fun_f {
            MakeIterator m_make_iterator;

            template <class Ptr, class Strides>
            GT_FUNCTION_DEVICE void operator()(Ptr &ptr, Strides const &strides) const {
                StencilStage()(m_make_iterator(), ptr, strides);
            }
        };

        template <class Sizes>
        bool is_domain_empty(const Sizes &sizes) {
            return tuple_util::host::apply([](auto... sizes) { return ((sizes == 0) || ...); }, sizes);
        }

        template <class ThreadBlockSizes,
            class LoopBlockSizes,
            class Sizes,
            class StencilStage,
            class MakeIterator,
            class Composite>
        void apply_stencil_stage(gpu<ThreadBlockSizes, LoopBlockSizes> const &g,
            Sizes const &sizes,
            StencilStage,
            MakeIterator make_iterator,
            Composite &&composite) {

            if (is_domain_empty(sizes)) {
                return;
            }

            auto ptr_holder = sid::get_origin(std::forward<Composite>(composite));
            auto strides = sid::get_strides(std::forward<Composite>(composite));

            auto [blocks, threads] = blocks_and_threads<ThreadBlockSizes, LoopBlockSizes>(sizes);
            assert(threads.x > 0 && threads.y > 0 && threads.z > 0);
            cuda_util::launch(blocks,
                threads,
                0,
                g.stream,
                kernel<ThreadBlockSizes,
                    LoopBlockSizes,
                    Sizes,
                    decltype(ptr_holder),
                    decltype(strides),
                    stencil_fun_f<StencilStage, MakeIterator>>,
                sizes,
                ptr_holder,
                strides,
                stencil_fun_f<StencilStage, MakeIterator>{std::move(make_iterator)});
        }

        template <class ColumnStage, class MakeIterator, class Seed>
        struct column_fun_f {
            MakeIterator m_make_iterator;
            Seed m_seed;
            int m_v_size;

            template <class Ptr, class Strides>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides) const {
                ColumnStage()(m_seed, m_v_size, m_make_iterator(), std::move(ptr), strides);
            }
        };

        template <class ThreadBlockSizes,
            class LoopBlockSizes,
            class Sizes,
            class ColumnStage,
            class MakeIterator,
            class Composite,
            class Vertical,
            class Seed>
        void apply_column_stage(gpu<ThreadBlockSizes, LoopBlockSizes> const &g,
            Sizes const &sizes,
            ColumnStage,
            MakeIterator make_iterator,
            Composite &&composite,
            Vertical,
            Seed seed) {

            if (is_domain_empty(sizes)) {
                return;
            }

            auto ptr_holder = sid::get_origin(std::forward<Composite>(composite));
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            auto h_sizes = hymap::canonicalize_and_remove_key<Vertical>(sizes);
            int v_size = at_key<Vertical>(sizes);

            auto [blocks, threads] = blocks_and_threads<ThreadBlockSizes, LoopBlockSizes>(h_sizes);
            assert(threads.x > 0 && threads.y > 0 && threads.z > 0);
            cuda_util::launch(blocks,
                threads,
                0,
                g.stream,
                kernel<ThreadBlockSizes,
                    LoopBlockSizes,
                    decltype(h_sizes),
                    decltype(ptr_holder),
                    decltype(strides),
                    column_fun_f<ColumnStage, MakeIterator, Seed>>,
                h_sizes,
                ptr_holder,
                strides,
                column_fun_f<ColumnStage, MakeIterator, Seed>{std::move(make_iterator), std::move(seed), v_size});
        }

        template <class ThreadBlockSizes, class LoopBlockSizes>
        auto tmp_allocator(gpu<ThreadBlockSizes, LoopBlockSizes> be) {
            return std::make_tuple(be, sid::device::cached_allocator(&cuda_util::cuda_malloc<char[]>));
        }

        template <class ThreadBlockSizes, class LoopBlockSizes, class Allocator, class Sizes, class T>
        auto allocate_global_tmp(
            std::tuple<gpu<ThreadBlockSizes, LoopBlockSizes>, Allocator> &alloc, Sizes const &sizes, data_type<T>) {
            return sid::make_contiguous<T, int_t, sid::unknown_kind>(std::get<1>(alloc), sizes);
        }
    } // namespace gpu_impl_

    using gpu_impl_::gpu;

    using gpu_impl_::apply_column_stage;
    using gpu_impl_::apply_stencil_stage;

    using gpu_impl_::allocate_global_tmp;
    using gpu_impl_::tmp_allocator;
} // namespace gridtools::fn::backend
