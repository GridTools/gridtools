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

    template <class BackendId>
    struct backend : public backend_base<BackendId> {
        typedef backend_base<BackendId> base_t;

        using typename base_t::backend_traits_t;
    };

} // namespace gridtools
