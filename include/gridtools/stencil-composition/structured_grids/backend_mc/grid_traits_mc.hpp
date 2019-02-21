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
#include "../../backend_ids.hpp"
#include "../../grid_traits_fwd.hpp"
#include "./execute_kernel_functor_mc_fwd.hpp"

namespace gridtools {
    template <class Args>
    struct kernel_functor_executor<backend_ids<target::mc, grid_type::structured, strategy::block>, Args> {
        using type = strgrid::execute_kernel_functor_mc<Args>;
    };
} // namespace gridtools
