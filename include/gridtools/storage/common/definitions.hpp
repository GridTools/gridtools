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

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    enum class ownership { external_gpu, external_cpu };
    enum class access_mode { read_write = 0, read_only = 1 };

    /**
     * @}
     */
} // namespace gridtools
