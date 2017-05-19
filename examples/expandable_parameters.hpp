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
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, enumtype::GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Naive >
#endif
#endif

namespace test_expandable_parameters {

    using namespace gridtools;
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct functor_exp {

#ifdef REASSIGN_DOMAIN
        typedef accessor< 0, enumtype::inout > parameters_out;
        typedef accessor< 1, enumtype::in > parameters_in;
#else
        typedef vector_accessor< 0, enumtype::inout > parameters_out;
        typedef vector_accessor< 1, enumtype::in > parameters_in;
#endif

        typedef boost::mpl::vector< parameters_out, parameters_in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(parameters_out{}) = eval(parameters_in{});
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

        typedef BACKEND::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        storage_t storage1(meta_data_, 1., "storage1");
        storage_t storage2(meta_data_, 2., "storage2");
        storage_t storage3(meta_data_, 3., "storage3");
        storage_t storage4(meta_data_, 4., "storage4");
        storage_t storage5(meta_data_, 5., "storage5");

        storage_t storage10(meta_data_, -1., "storage10");
        storage_t storage20(meta_data_, -2., "storage20");
        storage_t storage30(meta_data_, -3., "storage30");
        storage_t storage40(meta_data_, -4., "storage40");
        storage_t storage50(meta_data_, -5., "storage50");

        std::vector< storage_t > list_out_ = {storage1, storage2, storage3, storage4, storage5};
        std::vector< storage_t > list_in_ = {storage10, storage20, storage30, storage40, storage50};

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        typedef arg< 0, std::vector< storage_t > > p_list_out;
        typedef arg< 1, std::vector< storage_t > > p_list_in;
        typedef tmp_arg< 2, std::vector< storage_t > > p_list_tmp;

        typedef boost::mpl::vector< p_list_out, p_list_in, p_list_tmp > args_t;

        aggregator_type< args_t > domain_(list_out_, list_in_);

        auto comp_ = make_computation< BACKEND >(expand_factor< 2 >(),
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                                                     define_caches(cache< IJ, local >(p_list_tmp())),
                                                     make_stage< functor_exp >(p_list_tmp(), p_list_in()),
                                                     make_stage< functor_exp >(p_list_out(), p_list_tmp())));

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

        bool success = true;
        for (uint_t l = 0; l < list_in_.size(); ++l) {
            auto inv = make_host_view(list_in_[l]);
            auto outv = make_host_view(list_out_[l]);
            assert(check_consistency(list_out_[l], outv) && "view cannot be used safely.");
            assert(check_consistency(list_in_[l], inv) && "view cannot be used safely.");
            for (uint_t i = 0; i < d1; ++i)
                for (uint_t j = 0; j < d2; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        if (inv(i, j, k) != outv(i, j, k)) {
                            std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                      << "in = " << inv(i, j, k) << ", out = " << outv(i, j, k) << std::endl;
                            success = false;
                        }
                    }
        }

        return success;
    }
} // namespace test_expandable_parameters
