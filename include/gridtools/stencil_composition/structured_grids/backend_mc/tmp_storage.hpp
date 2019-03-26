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

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"

namespace gridtools {
    namespace tmp_storage {
        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_i_size(target::mc const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<0>();
            static constexpr auto alignment = StorageInfo::alignment_t::value;
            return (block_size + 2 * halo + alignment - 1) / alignment * alignment;
        }

        template <class /*StorageInfo*/, class /*MaxExtent*/>
        GT_FUNCTION int_t get_i_block_offset(target::mc const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            return false ? 0 : throw "should not be used";
        }

        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_j_size(target::mc const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<1>();
            return (block_size + 2 * halo) * omp_get_max_threads();
        }

        template <class /*StorageInfo*/, class /*MaxExtent*/>
        GT_FUNCTION int_t get_j_block_offset(target::mc const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            return false ? 0 : throw "should not be used";
        }
    } // namespace tmp_storage
} // namespace gridtools
