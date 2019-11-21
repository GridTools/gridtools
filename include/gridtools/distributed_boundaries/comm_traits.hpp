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

#include "../common/layout_map.hpp"
#include "../communication/GCL.hpp"
#ifdef GCL_MPI
#include "../communication/low_level/proc_grids_3D.hpp"
#else
#include "./mock_pattern.hpp"
#endif

namespace gridtools {

#ifndef GCL_MPI
    using namespace mock_;
#endif

    /** \ingroup Distributed-Boundaries
     * @{ */

    template <typename StorageType, typename Arch>
    struct comm_traits {
        using proc_layout = layout_map<0, 1, 2>;
        using comm_arch_type = Arch;
        using data_layout = typename StorageType::element_type::layout_t;
        using value_type = typename StorageType::element_type::data_t;
    };

    /** @} */

} // namespace gridtools
