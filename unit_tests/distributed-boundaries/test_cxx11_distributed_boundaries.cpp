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

// -*- compile-command: "cd /scratch/snx3000/mbianco/gt_project/build/; /opt/cray/pe/craype/2.5.12/bin/CC
// -DBACKEND_BLOCK -DBENCHMARK -DBOOST_NO_CXX11_DECLTYPE -DBOOST_RESULT_OF_USE_TR1 -DENABLE_METERS -DFLOAT_PRECISION=8
// -DGTEST_COLOR -DSUPPRESS_MESSAGES -I/scratch/snx3000/mbianco/gt_project/tools/googletest/googletest
// -I/scratch/snx3000/mbianco/gt_project/tools/googletest/googletest/include
// -I/scratch/snx3000/mbianco/gt_project/include  -I/include -DFUSION_MAX_VECTOR_SIZE=20 -DFUSION_MAX_MAP_SIZE=20
// -DSTRUCTURED_GRIDS -isystem/users/vogtha/boost_1_63_0 -mtune=native -march=native --std=c++11 -fopenmp -D_GCL_MPI_ -g
// /scratch/snx3000/mbianco/gt_project/unit_tests/distributed_boundaries/distributed_boundaries.cpp
// -L/scratch/snx3000/mbianco/gt_project/build -lgcl  && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH; srun -C gpu ./a.out"
// -*-

#include <iomanip>
#include <mpi.h>

#include <distributed-boundaries/comm_traits.hpp>
#include <distributed-boundaries/distributed_boundaries.hpp>

#include <boundary-conditions/value.hpp>
#include <boundary-conditions/copy.hpp>

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

int main(int argc, char **argv) {

    gridtools::GCL_Init(argc, argv);

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
    using storage_type = storage_tr::data_store_t< float_type, storage_info_t >;

    const uint_t halo_size = 2;
    uint_t d1 = 6;
    uint_t d2 = 7;
    uint_t d3 = 2;

    halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};
    halo_descriptor dk{0, 0, 0, d3 - 1, d3};
    array< halo_descriptor, 3 > halos{di, dj, dk};

    storage_info_t storage_info(d1, d2, d3);

    storage_type a(storage_info, 1, "a");
    storage_type b(storage_info, 2, "b");
    storage_type c(storage_info, 3, "c");
    storage_type d(storage_info, 4, "b");

    show_view(make_host_view(a));
    show_view(make_host_view(b));
    show_view(make_host_view(c));
    show_view(make_host_view(d));

    using cabc_t = distributed_boundaries< comm_traits< storage_type, gcl_cpu > >;

    cabc_t cabc{halos, {false, false, false}, 4, GCL_WORLD};

    cabc.exchange(bind_bc(value_boundary< float_type >{7.5}, a), bind_bc(copy_boundary{}, b, c), d);
    std::cout << "**********************************************\n";
    std::cout << "**********************************************\n";
    std::cout << "**********************************************\n";
    std::cout << "**********************************************\n";
    std::cout << "**********************************************\n";

    show_view(make_host_view(a));
    show_view(make_host_view(b));
    show_view(make_host_view(c));
    show_view(make_host_view(d));

    GCL_Finalize();
}
