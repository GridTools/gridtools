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

#include <boost/mpl/vector.hpp>

namespace gridtools {
    /**
       This is a syntactic token which is used to declare the public
       interface of a stencil operator. This is used to define the
       tuple of arguments/accessors that a stencil operator expects.

       TODO: make it not a mpl::vector to allow the drop of boost::mpl

       \tparam list List of accessors that are the arguments of the stancil operator
     */
    template <typename... list>
    using make_param_list = boost::mpl::vector<list...>;
} // namespace gridtools

#ifdef GT_STRUCTURED_GRIDS
#include "./structured_grids/esf.hpp"
#else
#include "./icosahedral_grids/esf.hpp"
#endif
