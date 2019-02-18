/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../common/storage_info_interface.hpp"

namespace gridtools {
    /*
     * @brief The mc storage info implementation.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment (mc_storage_info is not aligned by default)
     */
    template <uint_t Id,
        typename Layout,
        typename Halo = zero_halo<Layout::masked_length>,
        typename Alignment = alignment<8>>
    using mc_storage_info = storage_info_interface<Id, Layout, Halo, Alignment>;
} // namespace gridtools
