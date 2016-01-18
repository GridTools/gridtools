#include <gridtools.hpp>
#include <common/defs.hpp>

#ifdef CXX11_ENABLED

#include "gtest/gtest.h"
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <stencil-composition/structured_grids/call_interfaces_metafunctions.hpp>
#include <type_traits>
#include <tuple>
#include <common/generic_metafunctions/v_item_to_vector.hpp>

struct pretent_aggregator {
    using value_type = double;

    template <typename Accessor>
    struct accessor_return_type {
        using type = double;
    };

    template <gridtools::uint_t I, gridtools::enumtype::intend Intent,
              typename Range, gridtools::ushort_t N>
    constexpr
    value_type operator()(gridtools::accessor<I,Intent, Range, N> ) const {
        return static_cast<value_type>(I+1000);
    }
};


struct pretent_function {
    typedef gridtools::accessor<0,  gridtools::enumtype::in> a0;
    typedef gridtools::accessor<1,  gridtools::enumtype::inout> a1;
    typedef gridtools::accessor<2,  gridtools::enumtype::in> a2;
    typedef gridtools::accessor<3,  gridtools::enumtype::inout> a3;

    template <typename Eval>
    static void Do(Eval const& eval) {
        eval(a1()) += eval(a0());
        eval(a3()) += eval(a2());
    }
};

template <typename... Args>
void complex_test(Args &... args)
{
    using namespace gridtools;

    using packtype =  typename _impl::package_args<Args...>::type;

    GRIDTOOLS_STATIC_ASSERT((std::is_same<typename std::tuple_element<0, std::tuple<Args...>>::type,
                             typename boost::mpl::at_c<packtype, 0>::type>::value), "0");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<typename _impl::wrap_reference
                             <typename std::tuple_element<1, std::tuple<Args...>>::type>,
                             typename boost::mpl::at_c<packtype, 1>::type>::value), "1");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<typename std::tuple_element<2, std::tuple<Args...>>::type,
                             typename boost::mpl::at_c<packtype, 2>::type>::value), "2");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<typename _impl::wrap_reference
                             <typename std::tuple_element<3, std::tuple<Args...>>::type>,
                             typename boost::mpl::at_c<packtype, 3>::type>::value), "3");

    typedef _impl::function_aggregator_procedure<
        pretent_aggregator,
        0,0,0,
        packtype
        > f_aggregator_t;

    GRIDTOOLS_STATIC_ASSERT((_impl::contains_value<
                             typename f_aggregator_t::non_accessor_indices,
                             boost::mpl::integral_c<int, 3>
                             >::type::value), "Contains 3");

    GRIDTOOLS_STATIC_ASSERT((_impl::contains_value<
                             typename f_aggregator_t::non_accessor_indices,
                             boost::mpl::integral_c<int, 1>
                             >::type::value), "Contains 1");

    GRIDTOOLS_STATIC_ASSERT((not _impl::contains_value<
                             typename f_aggregator_t::non_accessor_indices,
                             boost::mpl::integral_c<int, 0>
                             >::type::value), "Contains 0");

    GRIDTOOLS_STATIC_ASSERT((not _impl::contains_value<
                             typename f_aggregator_t::non_accessor_indices,
                             boost::mpl::integral_c<int, 2>
                             >::type::value), "Contains 2");

    auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

    pretent_function::Do
        (
         f_aggregator_t
         (
          pretent_aggregator(),
          y
          )
         );

}


TEST(call_interfaces_metafunctions, compile_time_basic_tests) {
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

}

TEST(call_interfaces_metafunctions, call_pretent_procedure) {
    using namespace gridtools;

    accessor<0, enumtype::in, extent<1,1,1,1>> a0;
    accessor<1, enumtype::inout> a2;
    double a1 = 3.14;
    double a3 = 666;

    complex_test(a0, a1, a2, a3);

    EXPECT_TRUE(a1 == 1003.14);
    EXPECT_TRUE(a3 == 1667);
}

#endif
