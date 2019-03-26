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

#include "../coordinate.hpp"

namespace gridtools {
    namespace tmp_storage {
        // Block specialisations
        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_i_size(target::x86 const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<coord_i<target::x86>::value>();
            return (block_size + 2 * halo) * omp_get_max_threads();
        }

        template <class StorageInfo, class /*MaxExtent*/>
        GT_FUNCTION int_t get_i_block_offset(target::x86 const &, uint_t block_size, uint_t /*block_no*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<coord_i<target::x86>::value>();
            return (block_size + 2 * halo) * omp_get_thread_num() + halo;
        }

        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_j_size(target::x86 const &, uint_t block_size, uint_t /*total_size*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<coord_j<target::x86>::value>();
            return block_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/>
        GT_FUNCTION int_t get_j_block_offset(target::x86 const &, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<coord_j<target::x86>::value>();
            return halo;
        }
    } // namespace tmp_storage
} // namespace gridtools
