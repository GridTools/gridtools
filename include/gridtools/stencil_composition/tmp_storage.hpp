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

/**
 *  @file
 *  API for the temporary storage allocation/offsets
 *
 *  Facade API:
 *    1. DataStore make_tmp_data_store<MaxExtent>(Backend, Arg, Grid);
 *    2. int_t get_tmp_storage_offset<StorageInfo, MaxExtent>(Backend, Strides, BlockIds, PositionsInBlock);
 *  where:
 *    MaxExtent - integral_constant with maximal absolute extent in I direction.
 *                TODO(anstaf): change to fully specified max extent
 *    Backend  - instantiation of backend
 *    Arg      - instantiation of arg
 *    Grid     - instantiation of grid
 *    Strides  - 3D struct with the strides that are taken from the DataStore, returned by make_tmp_data_store
 *    BlockIds - 3D struct that specifies the position of the block in the i,j,k directions
 *    PositionsInBlock - 3D struct that specifies the position of the target point within the block
 *
 *  Backend API:
 *    1. get_i_size, get_j_size, and optionally get_k_size
 *    2. get_i_block_offset, get_j_block_offset and optionally get_k_block_offset
 *    3. make_storage_info
 *
 *    Signatures:
 *    StorageInfo make_storage_info<StorageInfo, NColors>(uint_t i_size, uint_t j_size, uint_t k_size);
 *    uint_t get_i_size<StorageInfo, MaxExtent>(Backend, uint_t block_size, uint_t total_size);
 *    GT_FUNCTION int_t get_k_block_offset<StorageInfo, MaxExtent>(Backend, uint_t block_size, uint_t total_size);
 *
 *    Backend overloads should be defined in the gridtools::tmp_storage namespace
 *    TODO(anstaf): switch to ADL lookup mechanism
 *
 */

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../meta.hpp"
#include "./arg.hpp"
#include "./backend_cuda/tmp_storage.hpp"
#include "./backend_naive/tmp_storage.hpp"
#include "./backend_x86/tmp_storage.hpp"
#include "./block.hpp"
#include "./grid.hpp"
#include "./location_type.hpp"
#include "./pos3.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "./structured_grids/tmp_storage.hpp"
#else
#include "./icosahedral_grids/tmp_storage.hpp"
#endif

namespace gridtools {
    namespace tmp_storage {
        template <class /*StorageInfo*/, class /*MaxExtent*/, class Backend>
        uint_t get_k_size(Backend const &, uint_t /*block_size*/, uint_t total_size) {
            return total_size;
        }
        template <class /*StorageInfo*/, class /*MaxExtent*/, class Backend>
        GT_FUNCTION int_t get_k_block_offset(Backend const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            return 0;
        }

        template <class Backend>
        constexpr std::true_type needs_allocate_cached_tmp(Backend const &) {
            return {};
        }
    } // namespace tmp_storage

    template <class MaxExtent, class ArgTag, class DataStore, int_t I, ushort_t NColors, class Backend, class Grid>
    DataStore make_tmp_data_store(
        Backend const &backend, plh<ArgTag, DataStore, location_type<I, NColors>, true> const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using namespace tmp_storage;
        using storage_info_t = typename DataStore::storage_info_t;
        return {make_storage_info<storage_info_t, NColors>(backend,
            get_i_size<storage_info_t, MaxExtent>(
                backend, block_i_size(backend, grid), grid.i_high_bound() - grid.i_low_bound() + 1),
            get_j_size<storage_info_t, MaxExtent>(
                backend, block_j_size(backend, grid), grid.j_high_bound() - grid.j_low_bound() + 1),
            get_k_size<storage_info_t, MaxExtent>(backend, block_k_size(backend, grid), grid.k_total_length()))};
    }

    template <class StorageInfo, class MaxExtent, class Backend, class Stride, class BlockNo, class PosInBlock>
    GT_FUNCTION int_t get_tmp_storage_offset(Backend const &backend,
        Stride const &GT_RESTRICT stride,
        BlockNo const &GT_RESTRICT block_no,
        PosInBlock const &GT_RESTRICT pos_in_block) {
        using namespace tmp_storage;
        static constexpr auto block_size =
            make_pos3(block_i_size(Backend{}), block_j_size(Backend{}), block_k_size(Backend{}));
        return stride.i *
                   (get_i_block_offset<StorageInfo, MaxExtent>(backend, block_size.i, block_no.i) + pos_in_block.i) +
               stride.j *
                   (get_j_block_offset<StorageInfo, MaxExtent>(backend, block_size.j, block_no.j) + pos_in_block.j) +
               stride.k *
                   (get_k_block_offset<StorageInfo, MaxExtent>(backend, block_size.k, block_no.k) + pos_in_block.k);
    };

    template <class Backend>
    GT_META_DEFINE_ALIAS(
        needs_allocate_cached_tmp, meta::id, decltype(::gridtools::tmp_storage::needs_allocate_cached_tmp(Backend{})));

} // namespace gridtools
