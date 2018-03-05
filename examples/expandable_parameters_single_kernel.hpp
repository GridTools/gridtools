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
#undef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>
#include "backend_select.hpp"

namespace test_expandable_parameters {

    using namespace gridtools;
    using namespace expressions;

    struct functor_single_kernel {

        typedef accessor< 0, enumtype::inout > parameters1_out;
        typedef accessor< 1, enumtype::inout > parameters2_out;
        typedef accessor< 2, enumtype::inout > parameters3_out;
        typedef accessor< 3, enumtype::inout > parameters4_out;
        typedef accessor< 4, enumtype::inout > parameters5_out;

        typedef accessor< 5, enumtype::in > parameters1_in;
        typedef accessor< 6, enumtype::in > parameters2_in;
        typedef accessor< 7, enumtype::in > parameters3_in;
        typedef accessor< 8, enumtype::in > parameters4_in;
        typedef accessor< 9, enumtype::in > parameters5_in;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector< parameters1_out,
            parameters2_out,
            parameters3_out,
            parameters4_out,
            parameters5_out,
            parameters1_in,
            parameters2_in,
            parameters3_in,
            parameters4_in,
            parameters5_in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation eval) {
            eval(parameters1_out()) = eval(parameters1_in());
            eval(parameters2_out()) = eval(parameters2_in());
            eval(parameters3_out()) = eval(parameters3_in());
            eval(parameters4_out()) = eval(parameters4_in());
            eval(parameters5_out()) = eval(parameters5_in());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

        typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
        typedef backend_t::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

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

        auto grid_ = make_grid(d1, d2, d3);

        typedef arg< 0, storage_t > p_0_out;
        typedef arg< 1, storage_t > p_1_out;
        typedef arg< 2, storage_t > p_2_out;
        typedef arg< 3, storage_t > p_3_out;
        typedef arg< 4, storage_t > p_4_out;

        typedef arg< 5, storage_t > p_0_in;
        typedef arg< 6, storage_t > p_1_in;
        typedef arg< 7, storage_t > p_2_in;
        typedef arg< 8, storage_t > p_3_in;
        typedef arg< 9, storage_t > p_4_in;

        typedef tmp_arg< 10, storage_t > p_0_tmp;
        typedef tmp_arg< 11, storage_t > p_1_tmp;
        typedef tmp_arg< 12, storage_t > p_2_tmp;
        typedef tmp_arg< 13, storage_t > p_3_tmp;
        typedef tmp_arg< 14, storage_t > p_4_tmp;

        auto comp_ = make_computation< backend_t >(grid_,
            p_0_out{} = storage1,
            p_1_out{} = storage2,
            p_2_out{} = storage3,
            p_3_out{} = storage4,
            p_4_out{} = storage5,
            p_0_in{} = storage10,
            p_1_in{} = storage20,
            p_2_in{} = storage30,
            p_3_in{} = storage40,
            p_4_in{} = storage50,
            make_multistage(enumtype::execute< enumtype::forward >(),
                                                       define_caches(cache< IJ, cache_io_policy::local >(
                                                           p_0_tmp(), p_1_tmp(), p_2_tmp(), p_3_tmp(), p_4_tmp())),
                                                       make_stage< functor_single_kernel >(p_0_tmp(),
                                                           p_1_tmp(),
                                                           p_2_tmp(),
                                                           p_3_tmp(),
                                                           p_4_tmp(),
                                                           p_0_in(),
                                                           p_1_in(),
                                                           p_2_in(),
                                                           p_3_in(),
                                                           p_4_in()),
                                                       make_stage< functor_single_kernel >(p_0_out(),
                                                           p_1_out(),
                                                           p_2_out(),
                                                           p_3_out(),
                                                           p_4_out(),
                                                           p_0_tmp(),
                                                           p_1_tmp(),
                                                           p_2_tmp(),
                                                           p_3_tmp(),
                                                           p_4_tmp())));

        comp_.run();
        comp_.sync_all();

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
