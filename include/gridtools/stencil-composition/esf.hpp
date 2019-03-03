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

#include <tuple>

namespace gridtools {
    /**
       This is a syntactic token which is used to declare the public
       interface of a stencil operator. This is used to define the
       tuple of arguments/accessors that a stencil operator expects.
     */
    template <class... Ts>
    using make_param_list = std::tuple<Ts...>;
} // namespace gridtools

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "./structured_grids/esf.hpp"
#else
#include "./icosahedral_grids/esf.hpp"
#endif
