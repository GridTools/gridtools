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

#include <mpi.h>

namespace gridtools {
    namespace gcl_impl_ {
        inline int &pid_holder() {
            static int res;
            return res;
        }

        inline int &procs_holder() {
            static int res;
            return res;
        }
    } // namespace gcl_impl_

    inline auto GCL_world() { return MPI_COMM_WORLD; }

    inline int GCL_pid() { return gcl_impl_::pid_holder(); }

    inline int GCL_procs() { return gcl_impl_::procs_holder(); }

    inline void GCL_Init(int argc, char **argv) {
        int ready;
        MPI_Initialized(&ready);
        if (!ready)
            MPI_Init(&argc, &argv);
        MPI_Comm_rank(GCL_world(), &gcl_impl_::pid_holder());
        MPI_Comm_size(GCL_world(), &gcl_impl_::procs_holder());
    }

    inline void GCL_Finalize() { MPI_Finalize(); }
} // namespace gridtools
