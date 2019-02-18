/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
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
