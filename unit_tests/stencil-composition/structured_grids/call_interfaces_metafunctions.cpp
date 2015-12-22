#include <gridtools.hpp>
#include <common/defs.hpp>

#ifdef CXX11_ENABLED

#include "gtest/gtest.h"
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <stencil-composition/structured_grids/call_interfaces_metafunctions.hpp>
#include <type_traits>


struct pretent_aggregator {
    using value_type = double;

    template <typename T0, typename T1, typename T2>
    value_type operator()(T0, T1, T2) const {
        return 0;
    }
};


template <typename... Args>
void complex_test(Args const&... args)
{
    using namespace gridtools;

    typedef function_aggregator_procedure<
        pretent_aggregator,
        0,0,0,
        typename _impl::package_args<Args...>::type
        > f_aggregator_t;

    auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);
}


TEST(call_interfaces_metafunctions, compile_time_tests) {
    using namespace gridtools;

    unsigned int v = 666;
    auto x = _impl::make_wrap(v);

    GRIDTOOLS_STATIC_ASSERT((std::is_same<decltype(x), _impl::wrap_reference<unsigned int>>::value), "");

    x.value() = 999;
    EXPECT_TRUE(x.value() == 999);

    accessor<0, enumtype::in, extent<1,1,1,1>> a0;
    accessor<1, enumtype::inout> a2;
    float a1 = 3.14;
    int a3 = 666;

    using pack =  _impl::package_args<decltype(a0),
                                      decltype(a1),
                                      decltype(a2),
                                      decltype(a3)>::type;

    GRIDTOOLS_STATIC_ASSERT((std::is_same<decltype(a0), boost::mpl::at_c<pack, 0>::type>::value), "1");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<_impl::wrap_reference<decltype(a1)>, boost::mpl::at_c<pack, 1>::type>::value), "2");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<decltype(a2), boost::mpl::at_c<pack, 2>::type>::value), "3");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<_impl::wrap_reference<decltype(a3)>, boost::mpl::at_c<pack, 3>::type>::value), "4");


    complex_test(a0, a1, a2, a3);
}

#endif
