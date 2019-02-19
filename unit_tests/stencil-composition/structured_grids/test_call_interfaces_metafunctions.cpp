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
#include <gridtools/common/defs.hpp>

#include "gtest/gtest.h"

#include <gridtools/stencil-composition/esf.hpp>
#include <gridtools/stencil-composition/stencil-functions/call_interfaces.hpp>
#include <gridtools/stencil-composition/stencil-functions/call_interfaces_metafunctions.hpp>
#include <tuple>
#include <type_traits>

struct pretent_aggregator {
    using value_type = double;

    template <gridtools::uint_t I, gridtools::intent Intent, typename Range, gridtools::ushort_t N>
    constexpr value_type operator()(gridtools::accessor<I, Intent, Range, N>) const {
        return static_cast<value_type>(I + 1000);
    }
};

struct pretent_function {
    typedef gridtools::accessor<0, gridtools::intent::in> a0;
    typedef gridtools::accessor<1, gridtools::intent::inout> a1;
    typedef gridtools::accessor<2, gridtools::intent::in> a2;
    typedef gridtools::accessor<3, gridtools::intent::inout> a3;

    template <typename Eval>
    static void apply(Eval &eval) {
        eval(a1()) += eval(a0());
        eval(a3()) += eval(a2());
    }
};

template <typename... Args>
void complex_test(Args &... args) {
    using namespace gridtools;

    using packtype = typename _impl::package_args<Args...>::type;

    GT_STATIC_ASSERT((std::is_same<typename std::tuple_element<0, std::tuple<Args...>>::type,
                         typename boost::mpl::at_c<packtype, 0>::type>::value),
        "0");
    GT_STATIC_ASSERT(
        (std::is_same<typename _impl::wrap_reference<typename std::tuple_element<1, std::tuple<Args...>>::type>,
            typename boost::mpl::at_c<packtype, 1>::type>::value),
        "1");
    GT_STATIC_ASSERT((std::is_same<typename std::tuple_element<2, std::tuple<Args...>>::type,
                         typename boost::mpl::at_c<packtype, 2>::type>::value),
        "2");
    GT_STATIC_ASSERT(
        (std::is_same<typename _impl::wrap_reference<typename std::tuple_element<3, std::tuple<Args...>>::type>,
            typename boost::mpl::at_c<packtype, 3>::type>::value),
        "3");

    typedef _impl::function_aggregator_procedure_offsets<pretent_aggregator, 0, 0, 0, packtype> f_aggregator_t;

    auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

    pretent_aggregator pa;
    f_aggregator_t fa(pa, y);
    pretent_function::apply(fa);
}

TEST(call_interfaces_metafunctions, compile_time_basic_tests) {
    using namespace gridtools;

    unsigned int v = 666;
    auto x = _impl::make_wrap(v);

    GT_STATIC_ASSERT((std::is_same<decltype(x), _impl::wrap_reference<unsigned int>>::value), "");

    x.value() = 999;
    EXPECT_TRUE(x.value() == 999);

    accessor<0, intent::in, extent<1, 1, 1, 1>> a0;
    accessor<1, intent::inout> a2;
    float a1 = 3.14;
    int a3 = 666;

    using pack = _impl::package_args<decltype(a0), decltype(a1), decltype(a2), decltype(a3)>::type;

    GT_STATIC_ASSERT((std::is_same<decltype(a0), boost::mpl::at_c<pack, 0>::type>::value), "1");
    GT_STATIC_ASSERT((std::is_same<_impl::wrap_reference<decltype(a1)>, boost::mpl::at_c<pack, 1>::type>::value), "2");
    GT_STATIC_ASSERT((std::is_same<decltype(a2), boost::mpl::at_c<pack, 2>::type>::value), "3");
    GT_STATIC_ASSERT((std::is_same<_impl::wrap_reference<decltype(a3)>, boost::mpl::at_c<pack, 3>::type>::value), "4");
}

TEST(call_interfaces_metafunctions, call_pretent_procedure) {
    using namespace gridtools;

    accessor<0, intent::in, extent<1, 1, 1, 1>> a0;
    accessor<1, intent::inout> a2;
    double a1 = 3.14;
    double a3 = 666;

    complex_test(a0, a1, a2, a3);

    EXPECT_TRUE(a1 == 1003.14);
    EXPECT_TRUE(a3 == 1667);
}

struct actual_function {
    typedef gridtools::accessor<0, gridtools::intent::in> a0;
    typedef gridtools::accessor<1, gridtools::intent::inout> a1;
    typedef gridtools::accessor<2, gridtools::intent::in> a2;

    typedef gridtools::make_param_list<a0, a1, a2> param_list;
};

struct another_function {
    typedef gridtools::accessor<0, gridtools::intent::inout> out;
    typedef gridtools::accessor<1, gridtools::intent::in, gridtools::extent<-1, 1, -1, 1>> in;

    typedef gridtools::make_param_list<out, in> param_list;
};

struct non_function_swap {
    typedef gridtools::accessor<1, gridtools::intent::inout> out;
    typedef gridtools::accessor<0, gridtools::intent::inout> in;

    typedef gridtools::make_param_list<in, out> param_list;
};

struct another_non_function {

    typedef gridtools::accessor<0, gridtools::intent::inout> out;
    typedef gridtools::accessor<1, gridtools::intent::in, gridtools::extent<0, 1, 0, 0>> in;
    typedef gridtools::accessor<2, gridtools::intent::inout> lap;

    typedef gridtools::make_param_list<out, in, lap> param_list;
};

TEST(call_interfaces_metafunctions, check_if_function) {
    GT_STATIC_ASSERT((gridtools::_impl::can_be_a_function<actual_function>::value == true), "");
    GT_STATIC_ASSERT((gridtools::_impl::_get_index_of_first_non_const<actual_function>::value == 1), "");

    GT_STATIC_ASSERT((gridtools::_impl::can_be_a_function<another_function>::value == true), "");
    GT_STATIC_ASSERT((gridtools::_impl::_get_index_of_first_non_const<another_function>::value == 0), "");

    GT_STATIC_ASSERT((gridtools::_impl::can_be_a_function<non_function_swap>::value == false), "");
    GT_STATIC_ASSERT((gridtools::_impl::_get_index_of_first_non_const<non_function_swap>::value == 0), "");

    GT_STATIC_ASSERT((gridtools::_impl::can_be_a_function<another_non_function>::value == false), "");
    GT_STATIC_ASSERT((gridtools::_impl::_get_index_of_first_non_const<another_non_function>::value == 0), "");
}
