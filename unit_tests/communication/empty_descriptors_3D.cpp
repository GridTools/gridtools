/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <common/boollist.hpp>
#include <communication/GCL.hpp>
#include <communication/high-level/descriptors.hpp>
#include <communication/high-level/descriptors_dt.hpp>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

#define DIM 10

struct triple_t {
    int x, y, z;
    triple_t(int a, int b, int c) : x(a), y(b), z(c) {}
    triple_t() : x(-1), y(-1), z(-1) {}
};

std::ostream &operator<<(std::ostream &s, triple_t const &t) {
    return s << " (" << t.x << ", " << t.y << ", " << t.z << ") ";
}

bool operator==(triple_t const &a, triple_t const &b) { return (a.x == b.x && a.y == b.y && a.z == b.z); }

bool operator!=(triple_t const &a, triple_t const &b) { return !(a == b); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    gridtools::GCL_Init(argc, argv);

    triple_t *a = new triple_t[DIM * DIM * DIM];
    triple_t *b = new triple_t[DIM * DIM * DIM];
    triple_t *c = new triple_t[DIM * DIM * DIM];

    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::cout << pid << " " << nprocs << "\n";

    std::stringstream ss;
    ss << pid;

    std::string filename = "out" + ss.str() + ".txt";
    // filename[3] = '0'+pid;
    std::cout << filename << std::endl;
    std::ofstream file(filename.c_str());

    file << pid << "  " << nprocs << "\n";

    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(nprocs, 3, dims);
    int period[3] = {0, 0, 0};

    file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

    typedef gridtools::MPI_3D_process_grid_t< 3 > grid_type;
    gridtools::array< int, 3 > dimensions;
    dimensions[0] = dims[0];
    dimensions[1] = dims[1];
    dimensions[2] = dims[2];

    gridtools::hndlr_dynamic_ut< triple_t, grid_type, gridtools::Halo_Exchange_3D< grid_type > > hd(
        gridtools::boollist< 3 >(false, false, false), CartComm, &dimensions);

    hd.halo.add_halo(2, 2, 1, 3, 6, DIM);
    hd.halo.add_halo(1, 2, 1, 3, 6, DIM);
    hd.halo.add_halo(0, 3, 2, 4, 6, DIM);

    hd.allocate_buffers(3);

    int pi, pj, pk;
    hd.pattern().proc_grid().coords(pk, pj, pi);
    int PI, PJ, PK;
    hd.pattern().proc_grid().dims(PK, PJ, PI);

    file << "COORDINATES " << pi << ", " << pj << ", " << pk << std::endl;

    for (int ii = 3; ii <= 6; ++ii)
        for (int jj = 3; jj <= 6; ++jj)
            for (int kk = 4; kk <= 6; ++kk) {
                a[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] =
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk);
            }

    for (int ii = 3; ii <= 6; ++ii)
        for (int jj = 3; jj <= 6; ++jj)
            for (int kk = 4; kk <= 6; ++kk) {
                b[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] =
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk);
            }

    for (int ii = 3; ii <= 6; ++ii)
        for (int jj = 3; jj <= 6; ++jj)
            for (int kk = 4; kk <= 6; ++kk) {
                c[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] =
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk);
            }

    hd.pack(a, b, c);

    hd.exchange();

    hd.unpack(a, b, c);

    // CHECK!
    bool err = false;
    for (int ii = 3 - ((pi > 0) ? 2 : 0); ii <= 6 + ((pi < PI - 1) ? 1 : 0); ++ii)
        for (int jj = 3 - ((pj > 0) ? 2 : 0); jj <= 6 + ((pj < PJ - 1) ? 1 : 0); ++jj)
            for (int kk = 4 - ((pk > 0) ? 3 : 0); kk <= 6 + ((pk < PK - 1) ? 2 : 0); ++kk) {
                if (a[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] !=
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk)) {
                    err = true;
                    file << " A " << ii << ", " << jj << ", " << kk << ", "
                         << a[gridtools::access(kk, jj, ii, DIM, DIM, DIM)]
                         << " != " << triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 3 + 4 * pk) << "\n";
                }
            }

    for (int ii = 3 - ((pi > 0) ? 2 : 0); ii <= 6 + ((pi < PI - 1) ? 1 : 0); ++ii)
        for (int jj = 3 - ((pj > 0) ? 2 : 0); jj <= 6 + ((pj < PJ - 1) ? 1 : 0); ++jj)
            for (int kk = 4 - ((pk > 0) ? 3 : 0); kk <= 6 + ((pk < PK - 1) ? 2 : 0); ++kk) {
                if (b[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] !=
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk)) {
                    err = true;
                    file << " B " << ii << ", " << jj << ", " << kk << ", "
                         << b[gridtools::access(kk, jj, ii, DIM, DIM, DIM)]
                         << " != " << triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk) << "\n";
                }
            }

    for (int ii = 3 - ((pi > 0) ? 2 : 0); ii <= 6 + ((pi < PI - 1) ? 1 : 0); ++ii)
        for (int jj = 3 - ((pj > 0) ? 2 : 0); jj <= 6 + ((pj < PJ - 1) ? 1 : 0); ++jj)
            for (int kk = 4 - ((pk > 0) ? 3 : 0); kk <= 6 + ((pk < PK - 1) ? 2 : 0); ++kk) {
                if (c[gridtools::access(kk, jj, ii, DIM, DIM, DIM)] !=
                    triple_t(ii - 3 + 4 * pi, jj - 3 + 4 * pj, kk - 4 + 3 * pk)) {
                    err = true;
                    file << " C " << ii << ", " << jj << ", " << kk << ", "
                         << c[gridtools::access(kk, jj, ii, DIM, DIM, DIM)]
                         << " != " << triple_t(ii + 7 * pi, jj - 3 + 4 * pj, kk + DIM * pk) << "\n";
                }
            }

    std::cout << std::boolalpha << err << " (False is good)" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
