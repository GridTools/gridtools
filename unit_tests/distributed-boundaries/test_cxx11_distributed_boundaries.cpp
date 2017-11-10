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
    std::cout << "length:total : " << view.storage_info().length() << ":" << view.storage_info().total_length() << ", ";

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

TEST(DistributedBoundaries, Test) {

#ifdef __CUDACC__
    typedef gridtools::backend< gridtools::enumtype::Cuda,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block > hd_backend;
    typedef gridtools::storage_traits< gridtools::enumtype::Cuda > storage_tr;
#else
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

    halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};
    halo_descriptor dk{0, 0, 0, d3 - 1, d3};
    array< halo_descriptor, 3 > halos{di, dj, dk};

    storage_info_t storage_info(d1, d2, d3);

    using cabc_t = distributed_boundaries< comm_traits< storage_type, gcl_cpu > >;

    cabc_t cabc{halos, {false, false, false}, 4, GCL_WORLD};

    int pi, pj, pk;
    cabc.proc_grid().coords(pi, pj, pk);
    int PI, PJ, PK;
    cabc.proc_grid().dims(PI, PJ, PK);

    storage_type a(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * (int)d1 + 100, j + pj * (int)d2 + 100, k + pk * (int)d3 + 100};
        },
        "a");
    storage_type b(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * (int)d1 + 1000, j + pj * (int)d2 + 1000, k + pk * (int)d3 + 1000};
        },
        "b");
    storage_type c(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * (int)d1 + 10000, j + pj * (int)d2 + 10000, k + pk * (int)d3 + 10000};
        },
        "c");
    storage_type d(storage_info,
        [=](int i, int j, int k) {
            return triplet{i + pi * (int)d1 + 100000, j + pj * (int)d2 + 100000, k + pk * (int)d3 + 100000};
        },
        "b");

    // show_view(make_host_view(a));
    // show_view(make_host_view(b));
    // show_view(make_host_view(c));
    // show_view(make_host_view(d));

    cabc.exchange(bind_bc(value_boundary< triplet >{triplet{42, 42, 42}}, a), bind_bc(copy_boundary{}, b, c), d);

    // show_view(make_host_view(a));
    // show_view(make_host_view(b));
    // show_view(make_host_view(c));
    // show_view(make_host_view(d));

    bool ok = true;
    for (int i = pi * d1; i < (pi + 1) * d1; ++i) {
        for (int j = pj * d2; j < (pj + 1) * d2; ++j) {
            for (int k = pk * d3; k < (pk + 1) * d3; ++k) {
                if (i < halo_size or j < halo_size or i >= PI * d1 - halo_size or j >= PJ * d2 - halo_size) {
                    // At the border
                    ok = ok and make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3) == triplet{42, 42, 42};
                    if (make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3) != triplet{42, 42, 42}) {
                        std::cout << "------------> " << i << ", " << j << ", " << k << " " << i - pi * d1 << ", "
                                  << j - pj * d2 << ", " << k - pk * d3 << " "
                                  << make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3)
                                  << " == " << triplet{42, 42, 42} << "\n";
                    }
                    ok = ok and
                         make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3) ==
                             make_host_view(c)(i - pi * d1, j - pj * d2, k - pk * d3);
                    if (make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3) !=
                        make_host_view(c)(i - pi * d1, j - pj * d2, k - pk * d3)) {
                        std::cout << "------------> " << i << ", " << j << ", " << k << " " << i - pi * d1 << ", "
                                  << j - pj * d2 << ", " << k - pk * d3 << " "
                                  << make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3)
                                  << " == " << make_host_view(c)(i - pi * d1, j - pj * d2, k - pk * d3) << "\n";
                    }
                } else {
                    // In the core
                    ok = ok and
                         make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3) ==
                             triplet{i + pi * (int)d1 + 100, j + pj * (int)d2 + 100, k + pk * (int)d3 + 100};
                    if (make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3) !=
                        triplet{i + pi * (int)d1 + 100, j + pj * (int)d2 + 100, k + pk * (int)d3 + 100}) {
                        std::cout << "------------> " << i << ", " << j << ", " << k << " " << i - pi * d1 << ", "
                                  << j - pj * d2 << ", " << k - pk * d3 << " "
                                  << make_host_view(a)(i - pi * d1, j - pj * d2, k - pk * d3) << " == "
                                  << triplet{i + pi * (int)d1 + 100, j + pj * (int)d2 + 100, k + pk * (int)d3 + 100}
                                  << "\n";
                    }
                    ok = ok and
                         make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3) ==
                             triplet{i + pi * (int)d1 + 1000, j + pj * (int)d2 + 1000, k + pk * (int)d3 + 1000};
                    if (make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3) !=
                        triplet{i + pi * (int)d1 + 1000, j + pj * (int)d2 + 1000, k + pk * (int)d3 + 1000}) {
                        std::cout << "------------> " << i << ", " << j << ", " << k << " " << i - pi * d1 << ", "
                                  << j - pj * d2 << ", " << k - pk * d3 << " "
                                  << make_host_view(b)(i - pi * d1, j - pj * d2, k - pk * d3) << " == "
                                  << triplet{i + pi * (int)d1 + 1000, j + pj * (int)d2 + 1000, k + pk * (int)d3 + 1000}
                                  << "\n";
                    }
                }
            }
        }
    }

    EXPECT_TRUE(ok);
}
