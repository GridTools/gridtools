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

#include "../GCL.hpp"
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace gridtools {

    template <typename _grid_>
    struct has_communicator {
        typedef typename boost::is_same<typename _grid_::has_communicator, boost::true_type>::type type;
    };

#ifndef GCL_MPI
#define MPI_Comm int
#endif

    template <typename _grid_>
    MPI_Comm get_communicator(
        _grid_ const &g, typename boost::enable_if<typename has_communicator<_grid_>::type>::type * = 0) {
        return g.communicator();
    }

    template <typename _grid_>
    MPI_Comm get_communicator(
        _grid_ const &g, typename boost::disable_if<typename has_communicator<_grid_>::type>::type * = 0) {
        return gridtools::GCL_WORLD;
    }

#ifndef GCL_MPI
#undef MPI_Comm
#endif
} // namespace gridtools
