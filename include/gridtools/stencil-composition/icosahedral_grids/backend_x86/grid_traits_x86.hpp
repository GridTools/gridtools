/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../../common/defs.hpp"
#include "../../backend_ids.hpp"
#include "../../grid_traits_fwd.hpp"
#include "execute_kernel_functor_x86_fwd.hpp"

namespace gridtools {
    template <class Strategy, class Args>
    struct kernel_functor_executor<backend_ids<target::x86, grid_type::icosahedral, Strategy>, Args> {
        using type = icgrid::execute_kernel_functor_x86<Args>;
    };
} // namespace gridtools
