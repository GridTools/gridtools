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
    namespace gcl {
        namespace impl_ {
            inline int &pid_holder() {
                static int res;
                return res;
            }

            inline int &procs_holder() {
                static int res;
                return res;
            }
        } // namespace impl_

        inline auto world() { return MPI_COMM_WORLD; }

        inline int pid() { return impl_::pid_holder(); }

        inline int procs() { return impl_::procs_holder(); }

        inline void init(int argc = 1, char **argv = 0) {
            int ready;
            MPI_Initialized(&ready);
            if (!ready)
                MPI_Init(&argc, &argv);
            MPI_Comm_rank(world(), &impl_::pid_holder());
            MPI_Comm_size(world(), &impl_::procs_holder());
        }

        inline void finalize() { MPI_Finalize(); }
    } // namespace gcl
} // namespace gridtools
