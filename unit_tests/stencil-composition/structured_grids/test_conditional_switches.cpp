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
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/conditionals/case_.hpp>
#include <gridtools/stencil-composition/conditionals/switch_.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace test_conditional_switches {
    using namespace gridtools;

    template <uint_t Id>
    struct functor1 {

        typedef accessor<0, intent::inout> p_dummy;
        typedef accessor<1, intent::inout> p_dummy_tmp;

        typedef make_param_list<p_dummy, p_dummy_tmp> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(p_dummy()) += Id;
        }
    };

    template <uint_t Id>
    struct functor2 {

        typedef accessor<0, intent::inout> p_dummy;
        typedef accessor<1, intent::in> p_dummy_tmp;

        typedef make_param_list<p_dummy, p_dummy_tmp> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(p_dummy()) += Id;
        }
    };

    bool test() {

        bool p = true;
        auto cond_ = [&p]() { return p ? 0 : 5; };
        auto nested_cond_ = []() { return 1; };
        auto other_cond_ = [&p]() { return p ? 1 : 2; };

        auto grid_ = make_grid(1, 1, 2);

        typedef gridtools::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t>
            data_store_t;

        storage_info_t meta_data_(8, 8, 8);
        data_store_t dummy(meta_data_, 0.);
        typedef arg<0, data_store_t> p_dummy;
        typedef tmp_arg<1, data_store_t> p_dummy_tmp;

        auto comp_ = make_computation<backend_t>(grid_,
            p_dummy{} = dummy,
            make_multistage(execute::forward(),
                make_stage<functor1<0>>(p_dummy(), p_dummy_tmp()),
                make_stage<functor2<0>>(p_dummy(), p_dummy_tmp())),
            switch_(cond_,
                case_(0,
                    make_multistage(execute::forward(),
                        make_stage<functor1<1>>(p_dummy(), p_dummy_tmp()),
                        make_stage<functor2<1>>(p_dummy(), p_dummy_tmp()))),
                case_(5,
                    switch_(nested_cond_,
                        case_(1,
                            make_multistage(execute::forward(),
                                make_stage<functor1<2000>>(p_dummy(), p_dummy_tmp()),
                                make_stage<functor2<2000>>(p_dummy(), p_dummy_tmp()))),
                        default_(make_multistage(execute::forward(),
                            make_stage<functor1<3000>>(p_dummy(), p_dummy_tmp()),
                            make_stage<functor2<3000>>(p_dummy(), p_dummy_tmp()))))),
                default_(make_multistage(execute::forward(),
                    make_stage<functor1<7>>(p_dummy(), p_dummy_tmp()),
                    make_stage<functor2<7>>(p_dummy(), p_dummy_tmp())))),
            switch_(other_cond_,
                case_(2,
                    make_multistage(execute::forward(),
                        make_stage<functor1<10>>(p_dummy(), p_dummy_tmp()),
                        make_stage<functor2<10>>(p_dummy(), p_dummy_tmp()))),
                case_(1,
                    make_multistage(execute::forward(),
                        make_stage<functor1<20>>(p_dummy(), p_dummy_tmp()),
                        make_stage<functor2<20>>(p_dummy(), p_dummy_tmp()))),
                default_(make_multistage(execute::forward(),
                    make_stage<functor1<30>>(p_dummy(), p_dummy_tmp()),
                    make_stage<functor2<30>>(p_dummy(), p_dummy_tmp())))),
            make_multistage(execute::forward(),
                make_stage<functor1<400>>(p_dummy(), p_dummy_tmp()),
                make_stage<functor2<400>>(p_dummy(), p_dummy_tmp())));

        comp_.run();
        dummy.sync();
        bool result = make_host_view(dummy)(0, 0, 0) == 842;

        p = false;
        comp_.run();

        comp_.sync_bound_data_stores();
        return result && make_host_view(dummy)(0, 0, 0) == 5662;
    }
} // namespace test_conditional_switches

TEST(stencil_composition, conditional_switch) { EXPECT_TRUE(test_conditional_switches::test()); }
