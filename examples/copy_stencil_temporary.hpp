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
#pragma once

#include "copy_stencil.hpp"

/**
  @file
  This file implements a trivial acceptance test for temporary storages:
  it consists of two stages: a copy of a field to a temporary, and a copy from
  the temporary to the output. An offset is added to make things slightly more
  interesting
*/

namespace copy_stencil_temporary {

    using namespace copy_stencil;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor1 {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            // std::cout <<eval(out()) << " =1= " << eval(in()) << "\n ";

            // if(threadIdx.x==0 && threadIdx.y==0)
            //     printf("[%d, %d] address %x\n", blockIdx.x, blockIdx.y, &eval(out()));
            eval(out()) = eval(in());
        }
    };

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor2 {

        typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 >, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            // std::cout <<eval(out()) << " =2= " << eval(in()) << "\n ";
            eval(out()) = eval(in(0, 0, 0));
        }
    };

    struct verify_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            // std::cout <<eval(out()) << " =1= " << eval(in()) << "\n ";

            if (eval(out()) != eval(in()))
                printf("error, %f != %f\n", eval(in()), eval(out()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_type;
        typedef BACKEND::temporary_storage_type< float_type, meta_data_t >::type tmp_storage_type;

        meta_data_t meta_data_(d1, d2, d3);

        // Definition of the actual data fields that are used for input/output
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, float_type(-1.), "out");
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    in(i, j, k) = i + j + k;
                }

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;
        typedef arg< 2, tmp_storage_type > p_tmp;

        typedef boost::mpl::vector< p_in, p_out, p_tmp > accessor_list;
        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< copy_functor1 >(p_in(), p_tmp()),
                    gridtools::make_stage< copy_functor2 >(p_tmp(), p_out()),
                    gridtools::make_stage< verify_functor >(p_in(), p_out())));

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

        bool success = true;
        if (verify) {
            for (uint_t i = 0; i < d1; ++i) {
                for (uint_t j = 0; j < d2; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {
                        if (in(i, j, k) != out(i, j, k)) {
                            std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                      << "in = " << in(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                            success = false;
                        }
                    }
                }
            }
        }

        return success;
    }
} // namespace copy_stencil
