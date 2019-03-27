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
#include "../backend_base.hpp"
#include "../backend_fwd.hpp"

namespace gridtools {

    template <class Target>
    struct backend : public backend_base<Target> {
        typedef backend_base<Target> base_t;
    };

} // namespace gridtools
