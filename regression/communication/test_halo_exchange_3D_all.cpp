/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <fstream>
#include <gridtools/common/boollist.hpp>
#include <gridtools/communication/halo_exchange.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/mpi_unit_test_driver/check_flags.hpp>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>

#include "gtest/gtest.h"

#include "triplet.hpp"

#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>

#ifdef __CUDACC__
#include <gridtools/common/cuda_util.hpp>
#endif

namespace halo_exchange_3D_all {
    int pid;
    int nprocs;
    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

    struct timeval start_tv;
    struct timeval stop1_tv;
    struct timeval stop2_tv;
    struct timeval stop3_tv;
    double lapse_time1;
    double lapse_time2;
    double lapse_time3;

#define B_ADD 1
#define C_ADD 2

#ifdef __CUDACC__
    typedef gridtools::gcl_gpu arch_type;
#else
    typedef gridtools::gcl_cpu arch_type;
#endif

    template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
    bool run(ST &file,
        int DIM1,
        int DIM2,
        int DIM3,
        int H,
        triple_t<USE_DOUBLE> *_a,
        triple_t<USE_DOUBLE> *_b,
        triple_t<USE_DOUBLE> *_c) {

        typedef gridtools::layout_map<I1, I2, I3> layoutmap;

        array<triple_t<USE_DOUBLE>, layoutmap> a(_a, (DIM1 + 2 * H), (DIM2 + 2 * H), (DIM3 + 2 * H));
        array<triple_t<USE_DOUBLE>, layoutmap> b(_b, (DIM1 + 2 * H), (DIM2 + 2 * H), (DIM3 + 2 * H));
        array<triple_t<USE_DOUBLE>, layoutmap> c(_c, (DIM1 + 2 * H), (DIM2 + 2 * H), (DIM3 + 2 * H));

        /* Just an initialization */
        for (int ii = 0; ii < DIM1 + 2 * H; ++ii)
            for (int jj = 0; jj < DIM2 + 2 * H; ++jj) {
                for (int kk = 0; kk < DIM3 + 2 * H; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE>();
                    b(ii, jj, kk) = triple_t<USE_DOUBLE>();
                    c(ii, jj, kk) = triple_t<USE_DOUBLE>();
                }
            }

        /* The pattern type is defined with the layouts, data types and
           number of dimensions.

           The logical assumption done in the program is that 'i' is the
           first dimension (rows), 'j' is the second, and 'k' is the
           third. The first layout states that 'i' is the second dimension
           in order of strides, while 'j' is the first and 'k' is the third
           (just by looking at the initialization loops this shoule be
           clear).

           The second layout states that the first dimension in data ('i')
           identify also the first dimension in the communicator. Logically,
           moving on 'i' dimension from processot (p,q,r) will lead you
           logically to processor (p+1,q,r). The other dimensions goes as
           the others.
        */
        static const int version = gridtools::version_manual;

        typedef gridtools::halo_exchange_dynamic_ut<layoutmap,
            gridtools::layout_map<0, 1, 2>,
            triple_t<USE_DOUBLE>::data_type,
            gridtools::MPI_3D_process_grid_t<3>,
            arch_type,
            version>
            pattern_type;

        /* The pattern is now instantiated with the periodicities and the
           communicator. The periodicity of the communicator is
           irrelevant. Setting it to be periodic is the best choice, then
           GCL can deal with any periodicity easily.
        */
        pattern_type he(typename pattern_type::grid_type::period_type(per0, per1, per2), CartComm);

        /* Next we need to describe the data arrays in terms of halo
           descriptors (see the manual). The 'order' of registration, that
           is the index written within <.> follows the logical order of the
           application. That is, 0 is associated to 'i', '1' is
           associated to 'j', '2' to 'k'.
        */
        he.template add_halo<0>(H, H, H, DIM1 + H - 1, DIM1 + 2 * H);
        he.template add_halo<1>(H, H, H, DIM2 + H - 1, DIM2 + 2 * H);
        he.template add_halo<2>(H, H, H, DIM3 + H - 1, DIM3 + 2 * H);

        /* Pattern is set up. This must be done only once per pattern. The
           parameter must me greater or equal to the largest number of
           arrays updated in a single step.
        */
        he.setup(3);

        file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";
        file.flush();

        /* Data is initialized in the inner region of size DIM1xDIM2
         */
        for (int ii = H; ii < DIM1 + H; ++ii)
            for (int jj = H; jj < DIM2 + H; ++jj)
                for (int kk = H; kk < DIM3 + H; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE>(
                        ii - H + (DIM1)*coords[0], jj - H + (DIM2)*coords[1], kk - H + (DIM3)*coords[2]);
                    b(ii, jj, kk) = triple_t<USE_DOUBLE>(ii - H + (DIM1)*coords[0] + B_ADD,
                        jj - H + (DIM2)*coords[1] + B_ADD,
                        kk - H + (DIM3)*coords[2] + B_ADD);
                    c(ii, jj, kk) = triple_t<USE_DOUBLE>(ii - H + (DIM1)*coords[0] + C_ADD,
                        jj - H + (DIM2)*coords[1] + C_ADD,
                        kk - H + (DIM3)*coords[2] + C_ADD);
                }

        printbuff(file, a, DIM1 + 2 * H, DIM2 + 2 * H, DIM3 + 2 * H);

        /* This is self explanatory now
         */

#ifdef __CUDACC__
        file << "***** GPU ON *****\n";
        file.flush();

        triple_t<USE_DOUBLE> *gpu_a = 0;
        triple_t<USE_DOUBLE> *gpu_b = 0;
        triple_t<USE_DOUBLE> *gpu_c = 0;
        GT_CUDA_CHECK(
            cudaMalloc(&gpu_a, (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>)));
        GT_CUDA_CHECK(
            cudaMalloc(&gpu_b, (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>)));
        GT_CUDA_CHECK(
            cudaMalloc(&gpu_c, (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>)));

        GT_CUDA_CHECK(cudaMemcpy(gpu_a,
            a.ptr,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyHostToDevice));

        GT_CUDA_CHECK(cudaMemcpy(gpu_b,
            b.ptr,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyHostToDevice));

        GT_CUDA_CHECK(cudaMemcpy(gpu_c,
            c.ptr,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyHostToDevice));

        std::vector<triple_t<USE_DOUBLE>::data_type *> vect(3);
        vect[0] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(gpu_a);
        vect[1] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(gpu_b);
        vect[2] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(gpu_c);
#else
        std::vector<triple_t<USE_DOUBLE>::data_type *> vect(3);
        vect[0] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(a.ptr);
        vect[1] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(b.ptr);
        vect[2] = reinterpret_cast<triple_t<USE_DOUBLE>::data_type *>(c.ptr);
#endif
        MPI_Barrier(gridtools::GCL_WORLD);

        gettimeofday(&start_tv, nullptr);

#ifdef VECTOR_INTERFACE
        he.pack(vect);
#else
        he.pack(vect[0], vect[1], vect[2]);
#endif

        gettimeofday(&stop1_tv, nullptr);

        he.exchange();

        gettimeofday(&stop2_tv, nullptr);

#ifdef VECTOR_INTERFACE
        he.unpack(vect);
#else
        he.unpack(vect[0], vect[1], vect[2]);
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&stop3_tv, nullptr);

        lapse_time1 =
            ((static_cast<double>(stop1_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop1_tv.tv_usec)) -
                (static_cast<double>(start_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_tv.tv_usec))) *
            1000.0;

        lapse_time2 =
            ((static_cast<double>(stop2_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop2_tv.tv_usec)) -
                (static_cast<double>(stop1_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop1_tv.tv_usec))) *
            1000.0;

        lapse_time3 =
            ((static_cast<double>(stop3_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop3_tv.tv_usec)) -
                (static_cast<double>(stop2_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop2_tv.tv_usec))) *
            1000.0;

        MPI_Barrier(MPI_COMM_WORLD);
        file << "TIME PACK: " << lapse_time1 << std::endl;
        file << "TIME EXCH: " << lapse_time2 << std::endl;
        file << "TIME UNPK: " << lapse_time3 << std::endl;
        file << "TIME ALL : " << lapse_time1 + lapse_time2 + lapse_time3 << std::endl;

        file << "\n********************************************************************************\n";
        file.flush();

#ifdef __CUDACC__
        GT_CUDA_CHECK(cudaMemcpy(a.ptr,
            gpu_a,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyDeviceToHost));

        GT_CUDA_CHECK(cudaMemcpy(b.ptr,
            gpu_b,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyDeviceToHost));

        GT_CUDA_CHECK(cudaMemcpy(c.ptr,
            gpu_c,
            (DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H) * sizeof(triple_t<USE_DOUBLE>),
            cudaMemcpyDeviceToHost));

        GT_CUDA_CHECK(cudaFree(gpu_a));
        GT_CUDA_CHECK(cudaFree(gpu_b));
        GT_CUDA_CHECK(cudaFree(gpu_c));
#endif

        printbuff(file, a, DIM1 + 2 * H, DIM2 + 2 * H, DIM3 + 2 * H);

        int passed = true;

        /* Checking the data arrived correctly in the whole region
         */
        for (int ii = 0; ii < DIM1 + 2 * H; ++ii)
            for (int jj = 0; jj < DIM2 + 2 * H; ++jj)
                for (int kk = 0; kk < DIM3 + 2 * H; ++kk) {

                    triple_t<USE_DOUBLE> ta;
                    triple_t<USE_DOUBLE> tb;
                    triple_t<USE_DOUBLE> tc;
                    int tax, tay, taz;
                    int tbx, tby, tbz;
                    int tcx, tcy, tcz;

                    tax = modulus(ii - H + (DIM1)*coords[0], DIM1 * dims[0]);
                    tbx = modulus(ii - H + (DIM1)*coords[0], DIM1 * dims[0]) + B_ADD;
                    tcx = modulus(ii - H + (DIM1)*coords[0], DIM1 * dims[0]) + C_ADD;

                    tay = modulus(jj - H + (DIM2)*coords[1], DIM2 * dims[1]);
                    tby = modulus(jj - H + (DIM2)*coords[1], DIM2 * dims[1]) + B_ADD;
                    tcy = modulus(jj - H + (DIM2)*coords[1], DIM2 * dims[1]) + C_ADD;

                    taz = modulus(kk - H + (DIM3)*coords[2], DIM3 * dims[2]);
                    tbz = modulus(kk - H + (DIM3)*coords[2], DIM3 * dims[2]) + B_ADD;
                    tcz = modulus(kk - H + (DIM3)*coords[2], DIM3 * dims[2]) + C_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H))) {
                            tax = triple_t<USE_DOUBLE>().x();
                            tbx = triple_t<USE_DOUBLE>().x();
                            tcx = triple_t<USE_DOUBLE>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H))) {
                            tay = triple_t<USE_DOUBLE>().y();
                            tby = triple_t<USE_DOUBLE>().y();
                            tcy = triple_t<USE_DOUBLE>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H))) {
                            taz = triple_t<USE_DOUBLE>().z();
                            tbz = triple_t<USE_DOUBLE>().z();
                            tcz = triple_t<USE_DOUBLE>().z();
                        }
                    }

                    ta = triple_t<USE_DOUBLE>(tax, tay, taz).floor();
                    tb = triple_t<USE_DOUBLE>(tbx, tby, tbz).floor();
                    tc = triple_t<USE_DOUBLE>(tcx, tcy, tcz).floor();

                    if (a(ii, jj, kk) != ta) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << " -> "
                             << "a " << a(ii, jj, kk) << " != " << ta << "\n";
                    }

                    if (b(ii, jj, kk) != tb) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << " -> "
                             << "b " << b(ii, jj, kk) << " != " << tb << "\n";
                    }

                    if (c(ii, jj, kk) != tc) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << " -> "
                             << "c " << c(ii, jj, kk) << " != " << tc << "\n";
                    }
                }

        return passed;
    }
    /** Each process will hold a tile of size
        (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
        the H width border is the inner region of an hypothetical stencil
        computation whise halo width is H.
    */
    bool test(int DIM1, int DIM2, int DIM3, int H, int P0 = 0, int P1 = 0) {
        /* Here we compute the computing gris as in many applications
         */
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        std::cout << pid << " " << nprocs << "\n";

        std::stringstream ss;
        ss << pid;

        std::string filename = "out" + ss.str() + ".txt";

        std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << pid << "  " << nprocs << "\n";

        dims[0] = P0;
        dims[1] = P1;
        dims[2] = 0;
        MPI_Dims_create(nprocs, 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

        MPI_Cart_get(CartComm, 3, dims, period, coords);

        /* This example will exchange 3 data arrays at the same time with
           different values.
        */
        triple_t<USE_DOUBLE> *_a = new triple_t<USE_DOUBLE>[(DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H)];
        triple_t<USE_DOUBLE> *_b = new triple_t<USE_DOUBLE>[(DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H)];
        triple_t<USE_DOUBLE> *_c = new triple_t<USE_DOUBLE>[(DIM1 + 2 * H) * (DIM2 + 2 * H) * (DIM3 + 2 * H)];

        bool passed = true;

#ifdef BENCH
        for (int i = 0; i < BENCH; ++i) {
            file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
            file.flush();
            passed = passed and run<std::ostream, 0, 1, 2, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
            file.flush();
        }
#else
        file << "run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";

        file << "Permutation 0,2,1\n";

        file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";

        file << "Permutation 1,0,2\n";

        file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";

        file << "Permutation 1,2,0\n";

        file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";

        file << "Permutation 2,0,1\n";

        file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";

        file << "Permutation 2,1,0\n";

        file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, true>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();

        file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, false>(file, DIM1, DIM2, DIM3, H, _a, _b, _c);
        file.flush();
        file << "---------------------------------------------------\n";
#endif

        if (passed)
            file << "RESULT: PASSED!\n";
        else
            file << "RESULT: FAILED!\n";

        delete[] _a;
        delete[] _b;
        delete[] _c;

        return passed;
    }
} // namespace halo_exchange_3D_all

#ifdef STANDALONE
int main(int argc, char **argv) {

#ifdef GT_USE_GPU
    device_binding();
#endif

    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

    if (argc != 5) {
        std::cout << "Usage: test_halo_exchange_3D dimx dimy dimz dim_halo\n where args are integer sizes of the data "
                     "fields and halo width"
                  << std::endl;
        return 1;
    }
    int DIM1 = atoi(argv[1]);
    int DIM2 = atoi(argv[2]);
    int DIM3 = atoi(argv[3]);
    int H = atoi(argv[4]);

    halo_exchange_3D_all::test(DIM1, DIM2, DIM3, H);

    MPI_Finalize();
}
#else
TEST(Communication, test_halo_exchange_3D_all) {
    bool passed = halo_exchange_3D_all::test(12, 12, 12, 2, 2, 1);
    EXPECT_TRUE(passed);
}
#endif
