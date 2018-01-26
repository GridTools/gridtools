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

#include <iomanip>
#include <mpi.h>
#include "gtest/gtest.h"
#include <tools/mpi_unit_test_driver/device_binding.hpp>

#include <distributed-boundaries/comm_traits.hpp>
#include <distributed-boundaries/distributed_boundaries.hpp>

#include <boundary-conditions/value.hpp>
#include <boundary-conditions/copy.hpp>

#include "../tools/triplet.hpp"

template < typename View >
void show_view(View const &view) {
    std::cout << "--------------------------------------------\n";

    std::cout << "lenth-end<i> : " << view.template length< 0 >() << ":" << view.template total_length< 0 >() << ", ";
    std::cout << "lenth-end<j> : " << view.template length< 1 >() << ":" << view.template total_length< 1 >() << ", ";
    std::cout << "lenth-end<k> : " << view.template length< 2 >() << ":" << view.template total_length< 2 >()
              << std::endl;

    std::cout << "i : " << view.template total_begin< 0 >() << ":" << view.template total_end< 0 >() << ", ";
    std::cout << "j : " << view.template total_begin< 1 >() << ":" << view.template total_end< 1 >() << ", ";
    std::cout << "k : " << view.template total_begin< 2 >() << ":" << view.template total_end< 2 >() << std::endl;

    std::cout << "i : " << view.template begin< 0 >() << ":" << view.template end< 0 >() << ", ";
    std::cout << "j : " << view.template begin< 1 >() << ":" << view.template end< 1 >() << ", ";
    std::cout << "k : " << view.template begin< 2 >() << ":" << view.template end< 2 >() << std::endl;
    std::cout << "--------------------------------------------\n";

    for (int k = view.template total_begin< 2 >(); k <= view.template total_end< 2 >(); ++k) {
        for (int i = view.template total_begin< 0 >(); i <= view.template total_end< 0 >(); ++i) {
            for (int j = view.template total_begin< 1 >(); j <= view.template total_end< 1 >(); ++j) {
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
template < typename PGrid >
bool from_neighbor(int dir_i, int dir_j, int dir_k, PGrid const &pg) {
    return pg.proc(dir_i, dir_j, dir_k) != -1;
}

TEST(DistributedBoundaries, Test) {

#ifdef __CUDACC__
    using comm_arch = gridtools::gcl_gpu;
    typedef gridtools::backend< gridtools::enumtype::Cuda,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block > hd_backend;
    typedef gridtools::storage_traits< gridtools::enumtype::Cuda > storage_tr;
#else
    using comm_arch = gridtools::gcl_cpu;
#ifdef BACKEND_BLOCK
    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block > hd_backend;
#else
    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Naive > hd_backend;
#endif
    typedef gridtools::storage_traits< gridtools::enumtype::Host > storage_tr;
#endif

    using namespace gridtools;

    using storage_info_t = storage_tr::storage_info_t< 0, 3, halo< 2, 2, 0 > >;
    using storage_type = storage_tr::data_store_t< triplet, storage_info_t >;

    const uint_t halo_size = 2;
    uint_t d1 = 6;
    uint_t d2 = 7;
    uint_t d3 = 2;

    storage_info_t storage_info(d1, d2, d3);

    using cabc_t = distributed_boundaries< comm_traits< storage_type, comm_arch > >;

    halo_descriptor di{
        halo_size, halo_size, halo_size, d1 - halo_size - 1, (unsigned)storage_info.padded_length< 0 >()};
    halo_descriptor dj{
        halo_size, halo_size, halo_size, d2 - halo_size - 1, (unsigned)storage_info.padded_length< 1 >()};
    halo_descriptor dk{0, 0, 0, d3 - 1, (unsigned)storage_info.dim< 2 >()};
    array< halo_descriptor, 3 > halos{di, dj, dk};

    cabc_t cabc{halos, {false, false, false}, 3, GCL_WORLD};

    int pi, pj, pk;
    cabc.proc_grid().coords(pi, pj, pk);
    int PI, PJ, PK;
    cabc.proc_grid().dims(PI, PJ, PK);

    storage_type a(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= halo_size and j >= halo_size and i < d1 - halo_size and j < d2 - halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100}
                         : triplet{0, 0, 0};
        },
        "a");
    storage_type b(storage_info,
        [=](int i, int j, int k) {
            bool inner = i >= halo_size and j >= halo_size and i < d1 - halo_size and j < d2 - halo_size;
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
            bool inner = i >= halo_size and j >= halo_size and i < d1 - halo_size and j < d2 - halo_size;
            return inner ? triplet{i + pi * ((int)d1 - 2 * (int)halo_size) + 100000,
                               j + pj * ((int)d2 - 2 * (int)halo_size) + 100000,
                               k + pk * ((int)d3 - 2 * (int)halo_size) + 100000}
                         : triplet{0, 0, 0};
        },
        "d");

    using namespace std::placeholders;

    cabc.exchange(
        bind_bc(value_boundary< triplet >{triplet{42, 42, 42}}, a), bind_bc(copy_boundary{}, b, _1).associate(c), d);

    a.sync();
    b.sync();
    c.sync();
    d.sync();

    bool ok = true;
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                if ((i + pi * d1) < halo_size or (j + pj * d2) < halo_size or (i + pi * d1) >= PI * d1 - halo_size or
                    (j + pj * d2) >= PJ * d2 - halo_size) {
                    // At the border
                    if (from_neighbor(region(i, d1, halo_size),
                            region(j, d2, halo_size),
                            region(k, d3, halo_size),
                            cabc.proc_grid())) {
                        ok = ok and
                             make_host_view(a)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * (int)halo_size) + 100,
                                                               j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                               i + pk * (int)(d3 - 2 * halo_size) + 100};
                        if (make_host_view(a)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 100}) {
                            std::cout << "a comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k)
                                      << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                       j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                       i + pk * (int)(d3 - 2 * halo_size) + 100}
                                      << "\n";
                        }

                        ok = ok and
                             make_host_view(b)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                               j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                               i + pk * (int)(d3 - 2 * halo_size) + 1000};
                        if (make_host_view(b)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 1000}) {
                            std::cout << "b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k)
                                      << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                       j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                       i + pk * (int)(d3 - 2 * halo_size) + 1000}
                                      << "\n";
                        }

                        ok = ok and
                             make_host_view(d)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                               j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                               i + pk * (int)(d3 - 2 * halo_size) + 100000};
                        if (make_host_view(d)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                              j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                              i + pk * (int)(d3 - 2 * halo_size) + 100000}) {
                            std::cout << "b comm ----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(d)(i, j, k)
                                      << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                       j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                       i + pk * (int)(d3 - 2 * halo_size) + 100000}
                                      << "\n";
                        }

                    } else {
                        ok = ok and make_host_view(a)(i, j, k) == triplet{42, 42, 42};
                        if (make_host_view(a)(i, j, k) != triplet{42, 42, 42}) {
                            std::cout << "a==42 -----------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(a)(i, j, k) << " == " << triplet{42, 42, 42} << "\n";
                        }
                        ok = ok and make_host_view(b)(i, j, k) == make_host_view(c)(i, j, k);
                        if (make_host_view(b)(i, j, k) != make_host_view(c)(i, j, k)) {
                            std::cout << "b==c ------------> " << i << ", " << j << ", " << k << " "
                                      << make_host_view(b)(i, j, k) << " == " << make_host_view(c)(i, j, k) << "\n";
                        }
                    }
                } else {
                    // In the core
                    ok = ok and
                         make_host_view(a)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                           j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                           k + pk * (int)(d3 - 2 * halo_size) + 100};
                    if (make_host_view(a)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 100}) {
                        std::cout << "a==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(a)(i, j, k)
                                  << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100,
                                                   j + pj * (int)(d2 - 2 * halo_size) + 100,
                                                   k + pk * (int)(d3 - 2 * halo_size) + 100}
                                  << "\n";
                    }
                    ok = ok and
                         make_host_view(b)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                           j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                           k + pk * (int)(d3 - 2 * halo_size) + 1000};
                    if (make_host_view(b)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 1000}) {
                        std::cout << "b==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(b)(i, j, k)
                                  << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 1000,
                                                   j + pj * (int)(d2 - 2 * halo_size) + 1000,
                                                   k + pk * (int)(d3 - 2 * halo_size) + 1000}
                                  << "\n";
                    }
                    ok = ok and
                         make_host_view(d)(i, j, k) == triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                           j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                           k + pk * (int)(d3 - 2 * halo_size) + 100000};
                    if (make_host_view(d)(i, j, k) != triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                          j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                          k + pk * (int)(d3 - 2 * halo_size) + 100000}) {
                        std::cout << "d==x ------------> " << i << ", " << j << ", " << k << " "
                                  << make_host_view(d)(i, j, k)
                                  << " == " << triplet{i + pi * (int)(d1 - 2 * halo_size) + 100000,
                                                   j + pj * (int)(d2 - 2 * halo_size) + 100000,
                                                   k + pk * (int)(d3 - 2 * halo_size) + 100000}
                                  << "\n";
                    }
                }
            }
        }
    }

    EXPECT_TRUE(ok);

    EXPECT_THROW(cabc.exchange(a, b, c, d), std::runtime_error);
}
