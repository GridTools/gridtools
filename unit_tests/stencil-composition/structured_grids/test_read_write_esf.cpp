/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

namespace rw_test {
    struct test_functor {
        typedef accessor<0, intent::in, extent<0>> i0;
        typedef accessor<1, intent::inout> o0;
        typedef accessor<2, intent::in, extent<2>> i1;
        typedef accessor<3, intent::inout> o1;
        typedef accessor<4, intent::in, extent<3>> i2;
        typedef accessor<5, intent::inout> o2;
        typedef accessor<6, intent::in, extent<4>> i3;
        typedef accessor<7, intent::inout> o3;

        typedef make_param_list<i0, o0, i1, o1, i2, o2, i3, o3> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {}
    };
} // namespace rw_test

using namespace rw_test;

TEST(esf, read_write) {

    typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t<double, meta_data_t> cell_storage_type;

    typedef arg<0, cell_storage_type> pi0;
    typedef arg<1, cell_storage_type> po0;
    typedef arg<2, cell_storage_type> pi1;
    typedef arg<3, cell_storage_type> po1;
    typedef arg<4, cell_storage_type> pi2;
    typedef arg<5, cell_storage_type> po2;
    typedef arg<6, cell_storage_type> pi3;
    typedef arg<7, cell_storage_type> po3;

    typedef make_param_list<pi0, po0, pi1, po1, pi2, po2, pi3, po3> args;

    typedef esf_descriptor<test_functor, args> esf_t;

    GT_STATIC_ASSERT(is_accessor_readonly<test_functor::i0>::type::value, "");
    GT_STATIC_ASSERT(!is_accessor_readonly<test_functor::o0>::type::value, "");

    GT_STATIC_ASSERT(is_accessor_readonly<test_functor::i1>::type::value, "");
    GT_STATIC_ASSERT(!is_accessor_readonly<test_functor::o1>::type::value, "");

    GT_STATIC_ASSERT(is_accessor_readonly<test_functor::i2>::type::value, "");
    GT_STATIC_ASSERT(!is_accessor_readonly<test_functor::o2>::type::value, "");

    GT_STATIC_ASSERT(is_accessor_readonly<test_functor::i3>::type::value, "");
    GT_STATIC_ASSERT(!is_accessor_readonly<test_functor::o3>::type::value, "");

    GT_STATIC_ASSERT(!is_written<esf_t>::apply<static_int<0>>::type::value, "");
    GT_STATIC_ASSERT(is_written<esf_t>::apply<static_int<1>>::type::value, "");

    GT_STATIC_ASSERT(!is_written<esf_t>::apply<static_int<2>>::type::value, "");
    GT_STATIC_ASSERT(is_written<esf_t>::apply<static_int<3>>::type::value, "");

    GT_STATIC_ASSERT(!is_written<esf_t>::apply<static_int<4>>::type::value, "");
    GT_STATIC_ASSERT(is_written<esf_t>::apply<static_int<5>>::type::value, "");

    GT_STATIC_ASSERT(!is_written<esf_t>::apply<static_int<6>>::type::value, "");
    GT_STATIC_ASSERT(is_written<esf_t>::apply<static_int<7>>::type::value, "");

    EXPECT_TRUE(true);
}
