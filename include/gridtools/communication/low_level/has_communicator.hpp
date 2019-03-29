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

#include <type_traits>

#include "../GCL.hpp"

namespace gridtools {

#ifndef GCL_MPI
#define MPI_Comm int
#endif

    template <typename _grid_>
    MPI_Comm get_communicator(_grid_ const &g, typename std::enable_if<_grid_::has_communicator::value>::type * = 0) {
        return g.communicator();
    }

    template <typename _grid_>
    MPI_Comm get_communicator(_grid_ const &g, typename std::enable_if<!_grid_::has_communicator::value>::type * = 0) {
        return gridtools::GCL_WORLD;
    }

#ifndef GCL_MPI
#undef MPI_Comm
#endif
} // namespace gridtools
