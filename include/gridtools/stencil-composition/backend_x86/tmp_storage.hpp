/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
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

namespace gridtools {
    namespace tmp_storage {
        // strategy::naive specialisations
        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        uint_t get_i_size(
            backend_ids<target::x86, GridType, strategy::naive> const &, uint_t /*block_size*/, uint_t total_size) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_i<backend_ids<target::x86, GridType, strategy::naive>>::value>();
            return total_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        GT_FUNCTION int_t get_i_block_offset(
            backend_ids<target::x86, GridType, strategy::naive> const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_i<backend_ids<target::x86, GridType, strategy::naive>>::value>();
            return halo;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        uint_t get_j_size(
            backend_ids<target::x86, GridType, strategy::naive> const &, uint_t /*block_size*/, uint_t total_size) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_j<backend_ids<target::x86, GridType, strategy::naive>>::value>();
            return total_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        GT_FUNCTION int_t get_j_block_offset(
            backend_ids<target::x86, GridType, strategy::naive> const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_j<backend_ids<target::x86, GridType, strategy::naive>>::value>();
            return halo;
        }

        // Block specialisations
        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        uint_t get_i_size(
            backend_ids<target::x86, GridType, strategy::block> const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_i<backend_ids<target::x86, GridType, strategy::block>>::value>();
            return (block_size + 2 * halo) * omp_get_max_threads();
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        GT_FUNCTION int_t get_i_block_offset(
            backend_ids<target::x86, GridType, strategy::block> const &, uint_t block_size, uint_t /*block_no*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_i<backend_ids<target::x86, GridType, strategy::block>>::value>();
            return (block_size + 2 * halo) * omp_get_thread_num() + halo;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        uint_t get_j_size(
            backend_ids<target::x86, GridType, strategy::block> const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_j<backend_ids<target::x86, GridType, strategy::block>>::value>();
            return block_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/, class GridType>
        GT_FUNCTION int_t get_j_block_offset(
            backend_ids<target::x86, GridType, strategy::block> const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo =
                StorageInfo::halo_t::template at<coord_j<backend_ids<target::x86, GridType, strategy::block>>::value>();
            return halo;
        }
    } // namespace tmp_storage
} // namespace gridtools
