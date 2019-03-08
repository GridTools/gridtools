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

namespace gridtools {
    /* here we store the buckets data structure, first target is 2D
     */
    template <typename PROC_GRID, int ALIGN_SIZE = 1> // ALIGN_SIZE is the size of the types used. Need to specify
    // better what it is. It is needed to allow send of receive to
    // other types (like MPI types) to use to send data.
    // there migh be a mpi_type<8>::value to be MPI_DOUBLE and
    // mpi_type<8>::divisor as the value to divide the legnth of the
    // buffer in bytes to compute sizes correctly. this is just a
    // proposal.
    struct Halo_Exchange_2D {};

    template <typename PROC_GRID, int ALIGN_SIZE = 1>
    struct Halo_Exchange_3D {};
} // namespace gridtools
