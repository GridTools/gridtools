/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/hymap.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../meta.hpp"
#include "../../arg.hpp"
#include "../../dim.hpp"
#include "../../execution_types.hpp"
#include "../../grid.hpp"
#include "../../pos3.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"
#include "../dim.hpp"
#include "local_domain.hpp"

namespace gridtools {
    namespace x86 {
        template <class StorageInfo>
        int_t get_i_block_offset(int_t block_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::i::value>();
            return (block_size + 2 * halo) * omp_get_thread_num() + halo;
        }

        template <class StorageInfo>
        int_t get_j_block_offset() {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::j::value>();
            return halo;
        }

        template <class StorageInfo, class Stride, class PosInBlock>
        int_t get_tmp_storage_offset(Stride stride, PosInBlock pos_in_block) {
            return stride.i * (get_i_block_offset<StorageInfo>(GT_DEFAULT_TILE_I) + pos_in_block.i) +
                   stride.j * (get_j_block_offset<StorageInfo>() + pos_in_block.j) + stride.k * pos_in_block.k;
        };

        template <class StrideMaps, class PtrMap>
        struct initialize_index_f {
            StrideMaps const &m_stride_maps;
            pos3<int_t> const &m_begin;
            pos3<int_t> const &m_block_no;
            pos3<int_t> const &m_pos_in_block;
            PtrMap &m_ptr_map;

            template <class Arg, std::enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {
                using storage_info_t = typename Arg::data_store_t::storage_info_t;
                GT_STATIC_ASSERT(is_storage_info<storage_info_t>::value, GT_INTERNAL_ERROR);

                host_device::at_key<Arg>(m_ptr_map) += get_tmp_storage_offset<storage_info_t>(
                    make_pos3<int_t>(sid::get_stride_element<Arg, dim::i>(m_stride_maps),
                        sid::get_stride_element<Arg, dim::j>(m_stride_maps),
                        sid::get_stride_element<Arg, dim::k>(m_stride_maps)),
                    m_pos_in_block);
            }

            template <class Arg, std::enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            GT_FORCE_INLINE void operator()() const {

                auto &ptr = at_key<Arg>(m_ptr_map);

                sid::shift(ptr,
                    sid::get_stride_element<Arg, dim::i>(m_stride_maps),
                    m_begin.i + m_block_no.i * GT_DEFAULT_TILE_I + m_pos_in_block.i);
                sid::shift(ptr,
                    sid::get_stride_element<Arg, dim::j>(m_stride_maps),
                    m_begin.j + m_block_no.j * GT_DEFAULT_TILE_J + m_pos_in_block.j);
                sid::shift(ptr, sid::get_stride_element<Arg, dim::k>(m_stride_maps), m_begin.k + m_pos_in_block.k);
            }
        };

        template <class StrideMaps, class PtrMap>
        GT_FORCE_INLINE initialize_index_f<StrideMaps, PtrMap> initialize_index(StrideMaps const &stride_maps,
            pos3<int_t> const &begin,
            pos3<int_t> const &block_no,
            pos3<int_t> const &pos_in_block,
            PtrMap &ptr_map) {
            return {stride_maps, begin, block_no, pos_in_block, ptr_map};
        }

        template <class ExecutionType,
            class From,
            class To,
            class Stages,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<!meta::is_empty<Stages>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<From, To, Stages>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            int_t n = grid.count(From(), To());
            for (int_t i = 0; i < n; ++i) {
                for_each<Stages>([&](auto stage) { stage(ptr, strides); });
                sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
            }
        }

        template <class ExecutionType,
            class Level,
            class Stages,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<!meta::is_empty<Stages>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<Level, Level, Stages>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            for_each<Stages>([&](auto stage) { stage(ptr, strides); });
            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
        }

        template <class ExecutionType,
            class From,
            class To,
            class Stages,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<meta::is_empty<Stages>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<From, To, Stages>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType> * grid.count(From{}, To{}));
        }

        template <class LoopIntervals, class ExecutionType, class LocalDomain, class Grid, class ExecutionInfo>
        GT_FORCE_INLINE void mss_loop(
            LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

            using extent_t = get_extent_from_loop_intervals<LoopIntervals>;
            using from_t = meta::first<meta::first<LoopIntervals>>;
            using to_t = meta::second<meta::last<LoopIntervals>>;

            int_t k_first = grid.template value_at<from_t>();

            auto ptr = local_domain.m_ptr_holder();

            for_each_type<get_keys<typename LocalDomain::ptr_t>>(initialize_index(local_domain.m_strides,
                {grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
                {execution_info.bi, execution_info.bj, 0},
                {extent_t::iminus::value, extent_t::jminus::value, k_first - grid.k_min()},
                ptr));

            auto block_size_f = [](int_t total, int_t block_size, int_t block_no) {
                int_t n = (total + block_size - 1) / block_size;
                return block_no == n - 1 ? total - block_no * block_size : block_size;
            };
            int_t total_i = grid.i_size();
            int_t total_j = grid.j_size();
            int_t size_i = block_size_f(total_i, GT_DEFAULT_TILE_I, execution_info.bi) + extent_t::iplus::value -
                           extent_t::iminus::value;
            int_t size_j = block_size_f(total_j, GT_DEFAULT_TILE_J, execution_info.bj) + extent_t::jplus::value -
                           extent_t::jminus::value;

            auto i_loop = sid::make_loop<dim::i>(size_i);
            auto j_loop = sid::make_loop<dim::j>(size_j);
            auto k_loop = [&](auto ptr, auto const &strides) {
                for_each<LoopIntervals>(
                    [&](auto loop_interval) { execute_interval<ExecutionType>(loop_interval, grid, ptr, strides); });
            };

            i_loop(j_loop(k_loop))(ptr, local_domain.m_strides);
        }
    } // namespace x86
} // namespace gridtools
