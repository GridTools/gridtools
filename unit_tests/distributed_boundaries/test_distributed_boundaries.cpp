/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iomanip>

#ifdef GCL_MPI
#include <mpi.h>
#endif

#include <gtest/gtest.h>

#include <gridtools/boundary_conditions/copy.hpp>
#include <gridtools/boundary_conditions/value.hpp>
#include <gridtools/distributed_boundaries/comm_traits.hpp>
#include <gridtools/distributed_boundaries/distributed_boundaries.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>

#include "../tools/triplet.hpp"

template <typename View>
void show_view(View const &view) {
    std::cout << "--------------------------------------------\n";

    std::cout << "length-end<i> : " << view.template length<0>() << ":" << view.template total_length<0>() << ", ";
    std::cout << "length-end<j> : " << view.template length<1>() << ":" << view.template total_length<1>() << ", ";
    std::cout << "length-end<k> : " << view.template length<2>() << ":" << view.template total_length<2>() << std::endl;

    std::cout << "i : " << view.template total_begin<0>() << ":" << view.template total_end<0>() << ", ";
    std::cout << "j : " << view.template total_begin<1>() << ":" << view.template total_end<1>() << ", ";
    std::cout << "k : " << view.template total_begin<2>() << ":" << view.template total_end<2>() << std::endl;

    std::cout << "i : " << view.template begin<0>() << ":" << view.template end<0>() << ", ";
    std::cout << "j : " << view.template begin<1>() << ":" << view.template end<1>() << ", ";
    std::cout << "k : " << view.template begin<2>() << ":" << view.template end<2>() << std::endl;
    std::cout << "--------------------------------------------\n";

    for (int k = view.template total_begin<2>(); k <= view.template total_end<2>(); ++k) {
        for (int i = view.template total_begin<0>(); i <= view.template total_end<0>(); ++i) {
            for (int j = view.template total_begin<1>(); j <= view.template total_end<1>(); ++j) {
                std::cout << std::setw(7) << std::setprecision(3) << view(i, j, k) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------------------------\n";
}

// Returns the relative coordinates of a neighbor processor given the dimensions of a storage
int region(int index, int size, int halo_size) {
    if (index < halo_size) {
        return -1;
    } else if (index >= size - halo_size) {
        return 1;
    }
    return 0;
}

// Tells if the neighbor in the relative coordinate dir_ exist
template <typename PGrid>
bool from_neighbor(int dir_i, int dir_j, int dir_k, PGrid const &pg) {
    return pg.proc(dir_i, dir_j, dir_k) != -1;
}

TEST(DistributedBoundaries, AvoidCommunicationOnlyBoundary) {

#ifdef __CUDACC__
    using comm_arch = gridtools::gcl_gpu;
#else
    using comm_arch = gridtools::gcl_cpu;
#endif
    using storage_tr = gridtools::storage_traits<backend_t>;

    using namespace gridtools;

    using storage_info_t = storage_tr::storage_info_t<0, 3, halo<2, 2, 0>>;
    using storage_type = storage_tr::data_store_t<triplet, storage_info_t>;

    const uint_t halo_size = 2;
    uint_t d1 = 6;
    uint_t d2 = 7;
    uint_t d3 = 2;

    storage_info_t storage_info(d1, d2, d3);

    using cabc_t = distributed_boundaries<comm_traits<storage_type, comm_arch>>;

    halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, (unsigned)storage_info.padded_length<0>()};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, (unsigned)storage_info.padded_length<1>()};
    halo_descriptor dk{0, 0, 0, d3 - 1, (unsigned)storage_info.total_length<2>()};
    array<halo_descriptor, 3> halos{di, dj, dk};

#ifndef GCL_MPI
    {
        // If MPI is not defined, the communication cannot be periodic.
        // This allows testing without MPI
        EXPECT_THROW((cabc_t{halos, {false, true, false}, 3, GCL_WORLD}), std::runtime_error);
    }
#endif

#ifdef GCL_MPI
    int dims[3] = {0, 0, 0};

    MPI_Dims_create(PROCS, 3, dims);

    int period[3] = {1, 1, 1};

    MPI_Comm CartComm;

    MPI_Cart_create(GCL_WORLD, 3, dims, period, false, &CartComm);
#else
    MPI_Comm CartComm = GCL_WORLD;
#endif

    cabc_t cabc{halos, {false, false, false}, 3, CartComm};

    int pi, pj, pk;
    cabc.proc_grid().coords(pi, pj, pk);
    int PI, PJ, PK;
    cabc.proc_grid().dims(PI, PJ, PK);

    storage_type a(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100}
                         : triplet{0, 0, 0};
        },
        "a");
    storage_type b(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 1000,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 1000,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 1000}
                         : triplet{0, 0, 0};
        },
        "b");
    storage_type c(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 10000,
                j + pj * ((int)d2 - 2 * (int)halo_size) + 10000,
                k + pk * ((int)d3 - 2 * (int)halo_size) + 10000};
        },
        "c");
    storage_type d(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100000,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100000,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100000}
                         : triplet{0, 0, 0};
        },
        "d");

    using namespace std::placeholders;

    cabc.boundary_only(
        bind_bc(value_boundary<triplet>{triplet{42, 42, 42}}, a), bind_bc(copy_boundary{}, b, _1).associate(c), d);

    a.sync();
    b.sync();
    c.sync();
    d.sync();

    bool ok = true;
    for (int i = 0; i < (int)d1; ++i) {
        for (int j = 0; j < (int)d2; ++j) {
            for (int k = 0; k < (int)d3; ++k) {
                if ((i + pi * d1) < (int)halo_size or (j + pj * d2) < (int)halo_size or
                    (i + pi * d1) >= PI * d1 - (int)halo_size or (j + pj * d2) >= PJ * d2 - (int)halo_size) {
                    // At the border
                    if (from_neighbor(region(i, d1, halo_size),
                            region(j, d2, halo_size),
                            region(k, d3, halo_size),
                            cabc.proc_grid())) {
                        ok = ok and make_host_view(a)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(a)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "edge a comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }

                        ok = ok and make_host_view(b)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(b)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "edge b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }

                        ok = ok and make_host_view(d)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(d)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "edge b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(d)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }

                    } else {
                        ok = ok and make_host_view(a)(i, j, k) == triplet{42, 42, 42};
                        if (make_host_view(a)(i, j, k) != triplet{42, 42, 42}) {
                            std::cout << gridtools::PID << ": "
                                      << "edge a==42 -----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == " << triplet{42, 42, 42} << "\n";
                        }
                        ok = ok and make_host_view(b)(i, j, k) == make_host_view(c)(i, j, k);
                        if (make_host_view(b)(i, j, k) != make_host_view(c)(i, j, k)) {
                            std::cout << gridtools::PID << ": "
                                      << "edge b==c ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == " << make_host_view(c)(i, j, k) << "\n";
                        }
                    }
                } else {
                    // In the core
                    if (region(i, d1, halo_size) == 0 and region(j, d2, halo_size) == 0) { // core-core
                        ok = ok and make_host_view(a)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                                      k + pk * (int)(d3 - 2 * halo_size) + 100};
                        if (make_host_view(a)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                              k + pk * (int)(d3 - 2 * halo_size) + 100}) {
                            std::cout << gridtools::PID << ": "
                                      << "core a==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                             j + pj * (int)(d2 - 2 * halo_size) + 100,
                                             k + pk * (int)(d3 - 2 * halo_size) + 100}
                                      << "\n";
                        }
                        ok = ok and make_host_view(b)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                                      k + pk * (int)(d3 - 2 * halo_size) + 1000};
                        if (make_host_view(b)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                              k + pk * (int)(d3 - 2 * halo_size) + 1000}) {
                            std::cout << gridtools::PID << ": "
                                      << "core b==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                             j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                             k + pk * (int)(d3 - 2 * halo_size) + 1000}
                                      << "\n";
                        }
                        ok = ok and make_host_view(d)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                                      k + pk * (int)(d3 - 2 * halo_size) + 100000};
                        if (make_host_view(d)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                              k + pk * (int)(d3 - 2 * halo_size) + 100000}) {
                            std::cout << gridtools::PID << ": "
                                      << "core d==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(d)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                             j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                             k + pk * (int)(d3 - 2 * halo_size) + 100000}
                                      << "\n";
                        }
                    } else { // core boundary
                        ok = ok and make_host_view(a)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(a)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "core a==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }
                        ok = ok and make_host_view(b)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(b)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "core b==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }
                        ok = ok and make_host_view(d)(i, j, k) == triplet{0, 0, 0};
                        if (make_host_view(d)(i, j, k) != triplet{0, 0, 0}) {
                            std::cout << gridtools::PID << ": "
                                      << "core d==x ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(d)(i, j, k) << " == " << triplet{0, 0, 0} << "\n";
                        }
                    }
                }
            }
        }
    }

    EXPECT_TRUE(ok);

    // This should not throw
    cabc.boundary_only(a, b, c, d);
}

TEST(DistributedBoundaries, Test) {

#ifdef __CUDACC__
    using comm_arch = gridtools::gcl_gpu;
#else
    using comm_arch = gridtools::gcl_cpu;
#endif
    using storage_tr = gridtools::storage_traits<backend_t>;

    using namespace gridtools;

    using storage_info_t = storage_tr::storage_info_t<0, 3, halo<2, 2, 0>>;
    using storage_type = storage_tr::data_store_t<triplet, storage_info_t>;

    const uint_t halo_size = 2;
    uint_t d1 = 6;
    uint_t d2 = 7;
    uint_t d3 = 2;

    storage_info_t storage_info(d1, d2, d3);

    using cabc_t = distributed_boundaries<comm_traits<storage_type, comm_arch>>;

    halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, (unsigned)storage_info.padded_length<0>()};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, (unsigned)storage_info.padded_length<1>()};
    halo_descriptor dk{0, 0, 0, d3 - 1, (unsigned)storage_info.total_length<2>()};
    array<halo_descriptor, 3> halos{di, dj, dk};

#ifdef GCL_MPI
    int dims[3] = {0, 0, 0};

    MPI_Dims_create(PROCS, 3, dims);

    int period[3] = {1, 1, 1};

    MPI_Comm CartComm;

    MPI_Cart_create(GCL_WORLD, 3, dims, period, false, &CartComm);
#else
    MPI_Comm CartComm = GCL_WORLD;
#endif

    cabc_t cabc{halos, {false, false, false}, 3, CartComm};

    int pi, pj, pk;
    cabc.proc_grid().coords(pi, pj, pk);
    int PI, PJ, PK;
    cabc.proc_grid().dims(PI, PJ, PK);

    storage_type a(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100}
                         : triplet{0, 0, 0};
        },
        "a");
    storage_type b(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 1000,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 1000,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 1000}
                         : triplet{0, 0, 0};
        },
        "b");
    storage_type c(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 10000,
                j + pj * ((int)d2 - 2 * (int)halo_size) + 10000,
                k + pk * ((int)d3 - 2 * (int)halo_size) + 10000};
        },
        "c");
    storage_type d(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= (int)halo_size and j >= (int)halo_size and i < (int)d1 - (int)halo_size and
                         j < (int)d2 - (int)halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100000,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100000,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100000}
                         : triplet{0, 0, 0};
        },
        "d");

    using namespace std::placeholders;

    cabc.exchange(
        bind_bc(value_boundary<triplet>{triplet{42, 42, 42}}, a), bind_bc(copy_boundary{}, b, _1).associate(c), d);

    a.sync();
    b.sync();
    c.sync();
    d.sync();

    bool ok = true;
    for (int i = 0; i < (int)d1; ++i) {
        for (int j = 0; j < (int)d2; ++j) {
            for (int k = 0; k < (int)d3; ++k) {
                if ((i + pi * d1) < halo_size or (j + pj * d2) < halo_size or (i + pi * d1) >= PI * d1 - halo_size or
                    (j + pj * d2) >= PJ * d2 - halo_size) {
                    // At the border
                    if (from_neighbor(region(i, d1, halo_size),
                            region(j, d2, halo_size),
                            region(k, d3, halo_size),
                            cabc.proc_grid())) {
                        ok = ok and make_host_view(a)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * (int)halo_size) + 100,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                                      i + pk * (int)(d3 - 2 * halo_size) + 100};
                        if (make_host_view(a)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 100}) {
                            std::cout << gridtools::PID << ": "
                                      << "a comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                             j + pj * (int)(d2 - 2 * halo_size) + 100,
                                             i + pk * (int)(d3 - 2 * halo_size) + 100}
                                      << "\n";
                        }

                        ok = ok and make_host_view(b)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                                      i + pk * (int)(d3 - 2 * halo_size) + 1000};
                        if (make_host_view(b)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 1000}) {
                            std::cout << gridtools::PID << ": "
                                      << "b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                             j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                             i + pk * (int)(d3 - 2 * halo_size) + 1000}
                                      << "\n";
                        }

                        ok = ok and make_host_view(d)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                                      j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                                      i + pk * (int)(d3 - 2 * halo_size) + 100000};
                        if (make_host_view(d)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 100000}) {
                            std::cout << gridtools::PID << ": "
                                      << "b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(d)(i, j, k) << " == "
                                      << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                             j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                             i + pk * (int)(d3 - 2 * halo_size) + 100000}
                                      << "\n";
                        }

                    } else {
                        ok = ok and make_host_view(a)(i, j, k) == triplet{42, 42, 42};
                        if (make_host_view(a)(i, j, k) != triplet{42, 42, 42}) {
                            std::cout << gridtools::PID << ": "
                                      << "a==42 -----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == " << triplet{42, 42, 42} << "\n";
                        }
                        ok = ok and make_host_view(b)(i, j, k) == make_host_view(c)(i, j, k);
                        if (make_host_view(b)(i, j, k) != make_host_view(c)(i, j, k)) {
                            std::cout << gridtools::PID << ": "
                                      << "b==c ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == " << make_host_view(c)(i, j, k) << "\n";
                        }
                    }
                } else {
                    // In the core
                    ok = ok and make_host_view(a)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                                  j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                                  k + pk * (int)(d3 - 2 * halo_size) + 100};
                    if (make_host_view(a)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 100}) {
                        std::cout << gridtools::PID << ": "
                                  << "core a==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(a)(i, j, k) << " == "
                                  << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                         j + pj * (int)(d2 - 2 * halo_size) + 100,
                                         k + pk * (int)(d3 - 2 * halo_size) + 100}
                                  << "\n";
                    }
                    ok = ok and make_host_view(b)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                                  j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                                  k + pk * (int)(d3 - 2 * halo_size) + 1000};
                    if (make_host_view(b)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 1000}) {
                        std::cout << gridtools::PID << ": "
                                  << "core b==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(b)(i, j, k) << " == "
                                  << triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                         j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                         k + pk * (int)(d3 - 2 * halo_size) + 1000}
                                  << "\n";
                    }
                    ok = ok and make_host_view(d)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                                  j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                                  k + pk * (int)(d3 - 2 * halo_size) + 100000};
                    if (make_host_view(d)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 100000}) {
                        std::cout << gridtools::PID << ": "
                                  << "core d==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(d)(i, j, k) << " == "
                                  << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                         j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                         k + pk * (int)(d3 - 2 * halo_size) + 100000}
                                  << "\n";
                    }
                }
            }
        }
    }

    EXPECT_TRUE(ok);

    std::cout << cabc.print_meters() << std::endl;

    EXPECT_THROW(cabc.exchange(a, b, c, d), std::runtime_error);
}
