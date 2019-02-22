/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <fstream>
#include <gridtools/common/array.hpp>
#include <gridtools/common/boollist.hpp>
#include <gridtools/communication/all_to_all_halo.hpp>
#include <gridtools/communication/low-level/proc_grids_3D.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>
#include <iostream>
#include <sstream>
#include <stdlib.h>

/*
  If GT_TEST_ENABLE_OUTPUT macro is defined then output is produced in
  one file per MPI rank
*/

namespace test_all_to_all_halo_3D {
    template <typename STREAM, typename T>
    void print(STREAM &cout, std::vector<T> const &v, int n, int m, int l) {
        if ((n < 40) && (m < 40)) {
            cout << "---------------------------------------------------------------------------------------\n\n";
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    cout << "@" << gridtools::PID << "@ ";
                    for (int k = 0; k < l; ++k) {
                        cout << v[i * m * l + j * l + k] << " ";
                    }
                    cout << "\n";
                }
                cout << "\n\n";
            }
            cout << "---------------------------------------------------------------------------------------\n\n";
        }
    }

    bool test(int N, int H) {

#ifdef GT_TEST_ENABLE_OUTPUT
        std::stringstream ss;
        ss << gridtools::PID;

        std::string filename = "out" + ss.str() + ".txt";
        // filename[3] = '0'+pid;
        std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());
#endif

        typedef gridtools::array<gridtools::halo_descriptor, 3> halo_block;

        typedef gridtools::MPI_3D_process_grid_t<3> grid_type;

        gridtools::array<int, 3> dims{0, 0, 0};
        grid_type pgrid(gridtools::boollist<3>(true, true, true), MPI_COMM_WORLD, dims);

        gridtools::all_to_all_halo<int, grid_type> a2a(pgrid, gridtools::GCL_WORLD);

        int pi, pj, pk;
        int PI, PJ, PK;
        pgrid.coords(pi, pj, pk);
        pgrid.dims(PI, PJ, PK);

#ifdef GT_TEST_ENABLE_OUTPUT
        file << "@" << gridtools::PID << "@ PROC GRID SIZE   " << PI << "x" << PJ << "x" << PK << "\n";
        file << "@" << gridtools::PID << "@ PROC COORDINATES " << pi << "x" << pj << "x" << pk << "\n";
        file << "@" << gridtools::PID << "@ PARAMETER N      " << N << "\n";

        file.flush();
#endif

        std::vector<int> dataout(PI * N * PJ * N * PK * N);
        std::vector<int> datain((N + 2 * H) * (N + 2 * H) * (N + 2 * H));

#ifdef GT_TEST_ENABLE_OUTPUT
        file << "Address of data: " << (void *)(&(dataout[0])) << ", data in " << (void *)(&(datain[0])) << "\n";
#endif

        gridtools::array<int, 3> crds;

        if (gridtools::PID == 0) {
#ifdef GT_TEST_ENABLE_OUTPUT
            file << "INITIALIZING DATA TO SEND\n";
#endif
            halo_block send_block;

            for (int i = 0; i < PI; ++i) {
                for (int j = 0; j < PJ; ++j) {
                    for (int k = 0; k < PK; ++k) {

                        crds[0] = i;
                        crds[1] = j; // DECREASING STRIDES
                        crds[2] = k; // DECREASING STRIDES

                        // DECREASING STRIDES
                        send_block[0] = gridtools::halo_descriptor(0, 0, i * N, (i + 1) * N - 1, PI * N);
                        send_block[1] = gridtools::halo_descriptor(0, 0, j * N, (j + 1) * N - 1, PJ * N);
                        send_block[2] = gridtools::halo_descriptor(0, 0, k * N, (k + 1) * N - 1, N * PK);

                        a2a.register_block_to(&dataout[0], send_block, crds);
                    }
                }
            }
        }

        crds[0] = 0;
        crds[1] = 0; // DECREASING STRIDES
        crds[2] = 0; // DECREASING STRIDES

        // INCREASING STRIDES
        halo_block recv_block;
        recv_block[0] = gridtools::halo_descriptor(H, H, H, N + H - 1, N + 2 * H);
        recv_block[1] = gridtools::halo_descriptor(H, H, H, N + H - 1, N + 2 * H);
        recv_block[2] = gridtools::halo_descriptor(H, H, H, N + H - 1, N + 2 * H);

        a2a.register_block_from(&datain[0], recv_block, crds);

        for (int i = 0; i < PI * N; ++i)
            for (int j = 0; j < PJ * N; ++j)
                for (int k = 0; k < PK * N; ++k) {
                    dataout[i * (PJ * N) * (PK * N) + j * (PK * N) + k] = i * (PJ * N) * (PK * N) + j * (PK * N) + k;
                }
        for (int i = 0; i < N + 2 * H; ++i)
            for (int j = 0; j < N + 2 * H; ++j)
                for (int k = 0; k < N + 2 * H; ++k) {
                    datain[i * (N + 2 * H) * (N + 2 * H) + j * (N + 2 * H) + k] = 0;
                }

#ifdef GT_TEST_ENABLE_OUTPUT
        print(file, dataout, PI * N, PJ * N, PK * N);
        print(file, datain, N + 2 * H, N + 2 * H, N + 2 * H);
        file.flush();
#endif
        MPI_Barrier(gridtools::GCL_WORLD);

        a2a.setup();
        a2a.start_exchange();
        a2a.wait();

#ifdef GT_TEST_ENABLE_OUTPUT
        print(file, dataout, PI * N, PJ * N, PK * N);
        print(file, datain, N + 2 * H, N + 2 * H, N + 2 * H);
#endif
        bool correct = true;

        int stride0 = (N * PK * N * PJ);
        int stride1 = (N * PK);
        int offseti = pi * N;
        int offsetj = pj * N;
        int offsetk = pk * N;

#ifdef GT_TEST_ENABLE_OUTPUT
        file << "Accessing " << offseti << ", " << offsetj << ", " << offsetk << "\n";
#endif
        for (int i = H; i < N + H; ++i)
            for (int j = H; j < N + H; ++j)
                for (int k = H; k < N + H; ++k) {
                    if (dataout[(offseti + i - H) * stride0 + (offsetj + j - H) * stride1 + offsetk + k - H] !=
                        datain[i * (N + 2 * H) * (N + 2 * H) + j * (N + 2 * H) + k]) {
#ifdef GT_TEST_ENABLE_OUTPUT
                        file << "(" << i << "," << j << "," << k << ") (" << (i - H + N * pi) * N * PK * N * PJ << ","
                             << (j - H + N * pj) * N * PK << "," << (k - H + N * pk) << ") Expected "
                             << dataout[(offseti + i - H) * stride0 + (offsetj + j - H) * stride1 + offsetk + k - H]
                             << " got " << datain[i * (N + 2 * H) * (N + 2 * H) + j * (N + 2 * H) + k] << std::endl;
#endif
                        correct = false;
                    }
                }

#ifdef GT_TEST_ENABLE_OUTPUT
        file << "RESULT: ";
        if (correct) {
            file << "PASSED!\n";
        } else {
            file << "FAILED!\n";
        }

        file.flush();
        file.close();
#endif

        return correct;
    }
} // namespace test_all_to_all_halo_3D

#ifdef STANDALONE
int main(int argc, char **argv) {

#ifdef GT_USE_GPU
    device_binding();
#endif

    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

    if (argc != 3) {
        std::cout << "Usage: pass two arguments: tile size (edge) followed by the halo width\n";
        return 1;
    }

    int N, H;
    N = atoi(argv[1]);
    H = atoi(argv[2]);

    bool passed = test_all_to_all_halo_3D::test(N, H);

    MPI_Barrier(gridtools::GCL_WORLD);

    gridtools::GCL_Finalize();
}
#else
TEST(Communication, test_all_to_all_halo_3D) {
    bool passed = test_all_to_all_halo_3D::test(13, 6);
    EXPECT_TRUE(passed);
}
#endif
