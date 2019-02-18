/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

//#include <gridtools/stencil-composition/esf.hpp>
#include <gridtools/stencil-composition/conditionals/if_.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

template <gridtools::uint_t Id>
struct functor {

    typedef gridtools::accessor<0, gridtools::intent::inout> a0;
    typedef gridtools::accessor<1, gridtools::intent::in> a1;
    typedef gridtools::make_param_list<a0, a1> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

bool predicate() { return false; }

TEST(unfold_all, test) {

    using namespace gridtools;

    auto grid = make_grid(2, 2, 3);

    typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, meta_data_t> storage_t;

    typedef arg<0, storage_t> p0;
    typedef arg<1, storage_t> p1;

    auto mss1 = make_multistage(execute::forward(),
        make_stage<functor<0>>(p0(), p1()),
        make_stage<functor<1>>(p0(), p1()),
        make_stage<functor<2>>(p0(), p1()),
        make_independent(make_stage<functor<3>>(p0(), p1()),
            make_stage<functor<4>>(p0(), p1()),
            make_independent(make_stage<functor<5>>(p0(), p1()), make_stage<functor<6>>(p0(), p1()))));

    auto mss2 = make_multistage(execute::forward(),
        make_stage<functor<7>>(p0(), p1()),
        make_stage<functor<8>>(p0(), p1()),
        make_stage<functor<9>>(p0(), p1()),
        make_independent(make_stage<functor<10>>(p0(), p1()),
            make_stage<functor<11>>(p0(), p1()),
            make_independent(make_stage<functor<12>>(p0(), p1()), make_stage<functor<13>>(p0(), p1()))));

    make_computation<backend_t>(grid, if_(predicate, mss1, mss2));
}
