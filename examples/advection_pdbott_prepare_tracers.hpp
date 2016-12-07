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

#include <stencil-composition/stencil-composition.hpp>
#include "benchmarker.hpp"
#include <tools/verifier.hpp>

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
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > interval_t;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct prepare_tracers {
        using data = vector_accessor< 0, inout >;
        using data_nnow = vector_accessor< 1, in >;
        using rho = accessor< 2, in >;
        typedef boost::mpl::vector< data, data_nnow, rho > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, interval_t) {
            eval(data()) = eval(rho()) * eval(data_nnow());
        }
    };

    template < typename Storage1, typename Storage2, typename Storage3 >
    void reference(Storage1 const &in_, Storage2 const &rho_, Storage3 &out_) {
        for (int_t i = 0; i < in_.meta_data().template dim< 0 >(); ++i)
            for (int_t j = 0; j < in_.meta_data().template dim< 1 >(); ++j)
                for (int_t k = 0; k < in_.meta_data().template dim< 2 >(); ++k){
                    out_(i, j, k) = rho_(i, j, k) * in_(i, j, k);
                }
    }

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t_steps) {

        typedef BACKEND::storage_info< 23, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        std::vector< pointer< storage_t > > list_out_(20, new storage_t(meta_data_, 0., "a storage"));
        std::vector< pointer< storage_t > > list_in_(20, new storage_t(meta_data_, 0., "a storage"));
        storage_t rho(meta_data_, 1.1, "rho");

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        typedef arg< 0, std::vector< pointer< storage_t > > > p_list_out;
        typedef arg< 1, std::vector< pointer< storage_t > > > p_list_in;
        typedef arg< 2, storage_t > p_rho;
        typedef boost::mpl::vector< p_list_out, p_list_in, p_rho > args_t;

        aggregator_type< args_t > domain_(boost::fusion::make_vector(&list_out_, &list_in_, &rho));

        auto comp_ =
            make_computation< BACKEND >(expand_factor< 20 >(),
                domain_,
                grid_,
                make_multistage(enumtype::execute< enumtype::forward >(),
                                            make_stage< prepare_tracers >(p_list_out(), p_list_in(), p_rho())));

        comp_->ready();
        comp_->steady();
        comp_->run();

#ifdef BENCHMARK
        benchmarker::run(comp_, t_steps);
#endif
        comp_->finalize();

        verifier verif(1e-6);
        array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};
        bool result = true;

        for (int_t l = 0; l < 20; ++l) {
            storage_t s_ref_(meta_data_, 0., "ref storage");
            reference(*list_in_[l], rho, s_ref_);
            result = result && verif.verify(grid_, *(list_out_[l]), s_ref_, halos);
        }

        return result;
    }
}
