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

#include <stencil-composition/stencil-composition.hpp>
#include "benchmarker.hpp"

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
//                   strides   1  x  xy
//                      dims   x  y  z
typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

namespace adv_prepare_tracers {

    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > interval_t;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct prepare_tracers {
        typedef accessor< 0, inout > data0;
        typedef accessor< 1, inout > data1;
        typedef accessor< 2, inout > data2;
        typedef accessor< 3, inout > data3;
        typedef accessor< 4, inout > data4;
        typedef accessor< 5, inout > data5;
        typedef accessor< 6, in > data_nnow0;
        typedef accessor< 7, in > data_nnow1;
        typedef accessor< 8, in > data_nnow2;
        typedef accessor< 9, in > data_nnow3;
        typedef accessor< 10, in > data_nnow4;
        typedef accessor< 11, in > data_nnow5;
        typedef accessor< 12, in > rho;
        typedef boost::mpl::vector< data0,
            data1,
            data2,
            data3,
            data4,
            data5,
            data_nnow0,
            data_nnow1,
            data_nnow2,
            data_nnow3,
            data_nnow4,
            data_nnow5,
            rho > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, interval_t) {
            eval(data0()) = eval(rho()) * eval(data_nnow0());
            eval(data1()) = eval(rho()) * eval(data_nnow1());
            eval(data2()) = eval(rho()) * eval(data_nnow2());
            eval(data3()) = eval(rho()) * eval(data_nnow3());
            eval(data4()) = eval(rho()) * eval(data_nnow4());
            eval(data5()) = eval(rho()) * eval(data_nnow5());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t_steps) {

        typedef BACKEND::storage_info< 23, layout_t > meta_data_t;
        typedef typename BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        std::vector< pointer< storage_t > > list_out_(20, new storage_t(meta_data_, 0., "a storage"));
        std::vector< pointer< storage_t > > list_in_(20, new storage_t(meta_data_, 0., "a storage"));
        storage_t rho(meta_data_, 1.1, "rho");

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        typedef arg< 0, storage_t > p_out0;
        typedef arg< 1, storage_t > p_out1;
        typedef arg< 2, storage_t > p_out2;
        typedef arg< 3, storage_t > p_out3;
        typedef arg< 4, storage_t > p_out4;
        typedef arg< 5, storage_t > p_out5;
        typedef arg< 6, storage_t > p_in0;
        typedef arg< 7, storage_t > p_in1;
        typedef arg< 8, storage_t > p_in2;
        typedef arg< 9, storage_t > p_in3;
        typedef arg< 10, storage_t > p_in4;
        typedef arg< 11, storage_t > p_in5;
        typedef arg< 12, storage_t > p_rho;
        typedef boost::mpl::
            vector< p_out0, p_out1, p_out2, p_out3, p_out4, p_out5, p_in0, p_in1, p_in2, p_in3, p_in4, p_in5, p_rho >
                args_t;

        aggregator_type< args_t > domain_(boost::fusion::make_vector(&(*list_out_[0]),
            &(*list_out_[1]),
            &(*list_out_[2]),
            &(*list_out_[3]),
            &(*list_out_[4]),
            &(*list_out_[5]),
            &(*list_in_[0]),
            &(*list_in_[1]),
            &(*list_in_[2]),
            &(*list_in_[3]),
            &(*list_in_[4]),
            &(*list_in_[5]),
            &rho));
#ifdef __CUDACC__
        gridtools::stencil *comp_ =
#else
        boost::shared_ptr< gridtools::stencil > comp_ =
#endif
            make_computation< BACKEND >(domain_,
                grid_,
                make_multistage(enumtype::execute< enumtype::forward >(),
                                            make_stage< prepare_tracers >(p_out0(),
                                                p_out1(),
                                                p_out2(),
                                                p_out3(),
                                                p_out4(),
                                                p_out5(),
                                                p_in0(),
                                                p_in1(),
                                                p_in2(),
                                                p_in3(),
                                                p_in4(),
                                                p_in5(),
                                                p_rho())));

        comp_->ready();
        comp_->steady();
        comp_->run();

#ifdef BENCHMARK
        benchmarker::run(comp_, t_steps);
#endif
        comp_->finalize();

        return true;
    }
}
