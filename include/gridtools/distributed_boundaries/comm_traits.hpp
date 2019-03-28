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

#include "../common/defs.hpp"
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
        template <typename GCLArch, typename = void>
        struct compute_arch_of {
            using type = target::x86;
        };

        template <typename T>
        struct compute_arch_of<gcl_gpu, T> {
            using type = target::cuda;
        };

        using proc_layout = gridtools::layout_map<0, 1, 2>;
        using comm_arch_type = Arch;
        using compute_arch = typename compute_arch_of<comm_arch_type>::type;
        using data_layout = typename StorageType::storage_info_t::layout_t;
        using value_type = typename StorageType::data_t;
    };

    /** @} */

} // namespace gridtools
