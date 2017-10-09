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
#include <tools/verifier.hpp>

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
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

    struct prepare_tracers {
        using data = vector_accessor< 0, inout >;
        using data_nnow = vector_accessor< 1, in >;
        using rho = accessor< 2, in >;
        typedef boost::mpl::vector< data, data_nnow, rho > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(data()) = eval(rho()) * eval(data_nnow());
        }
    };

    template < typename Storage1, typename Storage2, typename Storage3 >
    void reference(Storage1 in_, Storage2 rho_, Storage3 out_) {
        auto inv = make_host_view(in_);
        auto outv = make_host_view(out_);
        auto rhov = make_host_view(rho_);
        for (int_t i = 0; i < in_.template unaligned_dim< 0 >(); ++i)
            for (int_t j = 0; j < in_.template unaligned_dim< 1 >(); ++j)
                for (int_t k = 0; k < in_.template unaligned_dim< 2 >(); ++k) {
                    outv(i, j, k) = rhov(i, j, k) * inv(i, j, k);
                }
    }

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t_steps, bool verify) {

        typedef BACKEND::storage_traits_t::storage_info_t< 23, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        constexpr uint_t vec_size = 11;

        meta_data_t meta_data_(d1, d2, d3);

        std::vector< storage_t > list_out_;
        std::vector< storage_t > list_in_;

        // fill expandable parameters
        for (unsigned i = 0; i < vec_size; ++i) {
            list_out_.push_back(storage_t(meta_data_, 0.0, "out"));
            list_in_.push_back(storage_t(meta_data_, i, "in"));
        }

        storage_t rho(meta_data_, 1.1);

        auto grid_ = make_grid(d1, d2, d3);

        typedef arg< 0, std::vector< storage_t > > p_list_out;
        typedef arg< 1, std::vector< storage_t > > p_list_in;
        typedef arg< 2, storage_t > p_rho;
        typedef boost::mpl::vector< p_list_out, p_list_in, p_rho > args_t;

        aggregator_type< args_t > domain_(list_out_, list_in_, rho);
        auto comp_ =
            make_computation< BACKEND >(expand_factor< 2 >(),
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

        for (int_t l = 0; l < vec_size; ++l) {
            storage_t s_ref_(meta_data_, 0.);
            reference(list_in_[l], rho, s_ref_);
            result = result && verif.verify(grid_, (list_out_[l]), s_ref_, halos);
        }

        return result;
    }
}
