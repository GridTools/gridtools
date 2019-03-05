/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/stencil_composition/conditionals/if_.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace test_conditionals {
    using namespace gridtools;

    using axis_t = axis<1>;
    using x_interval = axis_t::full_interval;

    template <uint_t Id>
    struct functor {

        typedef accessor<0, intent::inout> p_dummy;
        typedef make_param_list<p_dummy> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(p_dummy()) = +Id;
        }
    };

    bool test() {

        auto cond = []() { return false; };
        auto cond2 = []() { return true; };

        auto grid_ = make_grid((uint_t)2, (uint_t)2, axis_t((uint_t)3));

        typedef gridtools::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t>
            data_store_t;
        storage_info_t meta_data_(3, 3, 3);
        data_store_t dummy(meta_data_, 0.);
        typedef arg<0, data_store_t> p_dummy;

        auto comp_ = make_computation<backend_t>(grid_,
            p_dummy() = dummy,
            if_(cond,
                make_multistage(execute::forward(), make_stage<functor<0>>(p_dummy())),
                if_(cond2,
                    make_multistage(execute::forward(), make_stage<functor<1>>(p_dummy())),
                    make_multistage(execute::forward(), make_stage<functor<2>>(p_dummy())))));

        comp_.run();
        comp_.sync_bound_data_stores();
        return make_host_view(dummy)(0, 0, 0) == 1;
    }
} // namespace test_conditionals

TEST(stencil_composition, conditionals) { EXPECT_TRUE(test_conditionals::test()); }
