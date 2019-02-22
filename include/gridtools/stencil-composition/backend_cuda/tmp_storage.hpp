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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"

#include "../backend_ids.hpp"
#include "../coordinate.hpp"
#include "../extent.hpp"

namespace gridtools {
    namespace tmp_storage {
        namespace _impl {
            template <class StorageInfo,
                uint_t Alignment = StorageInfo::alignment_t::value ? StorageInfo::alignment_t::value : 1>
            GT_FUNCTION constexpr uint_t align(uint_t x) {
                return (x + Alignment - 1) / Alignment * Alignment;
            }
            template <class StorageInfo, class MaxExtent>
            GT_FUNCTION constexpr uint_t full_block_i_size(uint_t block_size) {
                return align<StorageInfo>(
                    block_size + static_cast<uint_t>(MaxExtent::iplus::value - MaxExtent::iminus::value));
            }
            template <class StorageInfo,
                class MaxExtent,
                class GridType,
                int_t UsedHalo = -MaxExtent::iminus::value,
                uint_t StorageHalo = StorageInfo::halo_t::template at<
                    coord_i<backend_ids<target::cuda, GridType, strategy::block>>::value>()>
            GT_FUNCTION constexpr uint_t additional_i_offset() {
                return StorageHalo > UsedHalo ? StorageHalo - UsedHalo : 0;
            }
        } // namespace _impl

        template <class GridType>
        constexpr std::false_type needs_allocate_cached_tmp(
            backend_ids<target::cuda, GridType, strategy::block> const &) {
            return {};
        }

        template <class StorageInfo, class MaxExtent, class GridType>
        uint_t get_i_size(
            backend_ids<target::cuda, GridType, strategy::block> const &, uint_t block_size, uint_t total_size) {
            GT_STATIC_ASSERT(is_extent<MaxExtent>::value, GT_INTERNAL_ERROR);
            static constexpr auto additional_offset = _impl::additional_i_offset<StorageInfo, MaxExtent, GridType>();
            auto full_block_size = _impl::full_block_i_size<StorageInfo, MaxExtent>(block_size);
            auto num_blocks = (total_size + block_size + 1) / block_size;
            return num_blocks * full_block_size + additional_offset;
        }

        template <class StorageInfo, class MaxExtent, class GridType>
        GT_FUNCTION int_t get_i_block_offset(
            backend_ids<target::cuda, GridType, strategy::block> const &, uint_t block_size, uint_t block_no) {
            GT_STATIC_ASSERT(is_extent<MaxExtent>::value, GT_INTERNAL_ERROR);
            static constexpr auto additional_offset = _impl::additional_i_offset<StorageInfo, MaxExtent, GridType>();
            auto full_block_size = _impl::full_block_i_size<StorageInfo, MaxExtent>(block_size);
            return static_cast<int_t>(block_no * full_block_size) - MaxExtent::iminus::value + additional_offset;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        uint_t get_j_size(
            backend_ids<target::cuda, GridType, strategy::block> const &, uint_t block_size, uint_t total_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<
                coord_j<backend_ids<target::cuda, GridType, strategy::block>>::value>();
            auto full_block_size = block_size + 2 * halo;
            auto num_blocks = (total_size + block_size - 1) / block_size;
            return full_block_size * num_blocks;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        GT_FUNCTION int_t get_j_block_offset(
            backend_ids<target::cuda, GridType, strategy::block> const &, uint_t block_size, uint_t block_no) {
            static constexpr auto halo = StorageInfo::halo_t::template at<
                coord_j<backend_ids<target::cuda, GridType, strategy::block>>::value>();
            return block_no * (block_size + 2 * halo) + halo;
        }
    } // namespace tmp_storage
} // namespace gridtools
