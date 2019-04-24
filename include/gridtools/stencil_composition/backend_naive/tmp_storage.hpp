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

#include "../dim.hpp"

namespace gridtools {
    namespace tmp_storage {
        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_i_size(backend::naive, uint_t /*block_size*/, uint_t total_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::i::value>();
            return total_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/>
        GT_FUNCTION int_t get_i_block_offset(backend::naive, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::i::value>();
            return halo;
        }

        template <class StorageInfo, class /*MaxExtent*/>
        uint_t get_j_size(backend::naive, uint_t /*block_size*/, uint_t total_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::j::value>();
            return total_size + 2 * halo;
        }

        template <class StorageInfo, class /*MaxExtent*/>
        GT_FUNCTION int_t get_j_block_offset(backend::naive, uint_t /*block_size*/, uint_t /*block_no*/) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::j::value>();
            return halo;
        }
    } // namespace tmp_storage
} // namespace gridtools
