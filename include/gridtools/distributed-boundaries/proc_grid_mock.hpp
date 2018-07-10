/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

namespace gridtools {
    namespace mock_ {
        using MPI_Comm = int;

        template <int D>
        struct MPI_3D_process_grid_t {

            MPI_3D_process_grid_t() {}
            template <typename A, typename B>
            MPI_3D_process_grid_t(A const &, B const &) {}
            MPI_3D_process_grid_t(MPI_3D_process_grid_t const &other) = default;
            template < typename A, typename B, typename C >
            MPI_3D_process_grid_t(A const &c, B const &comm, C const &) {}

            MPI_Comm communicator() const { return 0; }
            void dims(int &t_R, int &t_C, int &t_S) const {
                t_R = 1;
                t_C = 1;
                t_S = 1;
            }

            void coords(int &t_R, int &t_C, int &t_S) const {
                t_R = 0;
                t_C = 0;
                t_S = 0;
            }

            template < int I, int J, int K >
            int proc() const {
                return proc(I,J,K);
            }

            int proc(int I, int J, int K) const {
                int _coords[3];

                // periodicity is false by default in this mock
                if (I != 0 or J != 0 or K != 0) {
                    return -1;
                }
                return 0;
            }

            int pid() const {
                return 0;
            }
        };
    }
}
