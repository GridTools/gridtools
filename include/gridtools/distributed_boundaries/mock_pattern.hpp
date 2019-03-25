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

#include "../common/boollist.hpp"

namespace gridtools {
    namespace mock_ {
        using MPI_Comm = int;

        template <int D>
        struct MPI_3D_process_grid_t {

            MPI_3D_process_grid_t() {}
            template <typename Period, typename Comm>
            MPI_3D_process_grid_t(Period const &a, Comm const &) {}
            MPI_3D_process_grid_t(MPI_3D_process_grid_t const &other) = default;
            template <typename Period, typename Comm, typename Dims>
            MPI_3D_process_grid_t(Period const &c, Comm const &comm, Dims const &) {}

            MPI_Comm communicator() const { return 0; }
            void dims(int &t_R, int &t_C, int &t_S) const {
                t_R = 1;
                t_C = 1;
                t_S = 1;
            }

            template <typename A>
            void fill_dims(A &) const {}

            void coords(int &t_R, int &t_C, int &t_S) const {
                t_R = 0;
                t_C = 0;
                t_S = 0;
            }

            template <int I, int J, int K>
            int proc() const {
                return proc(I, J, K);
            }

            int proc(int I, int J, int K) const {
                // periodicity is false by default in this mock
                if (I != 0 or J != 0 or K != 0) {
                    return -1;
                }
                return 0;
            }

            int pid() const { return 0; }
        };

        struct pattern_t {
            MPI_3D_process_grid_t<3> m_comm;

            pattern_t(MPI_3D_process_grid_t<3> m_comm) : m_comm{m_comm} {}

            MPI_3D_process_grid_t<3> proc_grid() const { return m_comm; }
        };

        template <typename, typename, typename, typename, typename>
        struct halo_exchange_dynamic_ut {
            boollist<3> m_period;
            MPI_3D_process_grid_t<3> m_comm;

            template <typename A>
            halo_exchange_dynamic_ut(boollist<3> p, A) : m_period{p}, m_comm{p, 0} {
                if ((m_period.value(0) != false) or (m_period.value(1) != false) or (m_period.value(2) != false)) {
                    throw(std::runtime_error(
                        "To use distributed-boundaries without MPI the communication pattern should not be periodic"));
                }
            }

            MPI_3D_process_grid_t<3> const &comm() const { return m_comm; }

            template <int, typename... As>
            void add_halo(As...) {}

            void setup(uint_t){};

            pattern_t pattern() const { return m_comm; }

            void exchange() {}

            template <typename... As>
            void pack(As...) {}

            template <typename... As>
            void unpack(As...) {}
        };

    } // namespace mock_
} // namespace gridtools
