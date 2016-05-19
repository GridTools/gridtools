#include "gtest/gtest.h"
#include "common/defs.hpp"
#include "common/tuple.hpp"

using namespace gridtools;

template<int_t A> struct ex;

template<uint_t A> struct exu;


TEST(tuple, test_tuple) {

    constexpr tuple<int_t, short_t, uint_t> tup(-3,4,10);

    GRIDTOOLS_STATIC_ASSERT((static_int<tup.n_dimensions>::value == 3), "ERROR");
    ASSERT_TRUE((tup.template get<0>() == -3));
    ASSERT_TRUE((tup.template get<1>() == 4));
    ASSERT_TRUE((tup.template get<2>() == 10));


    //make sure the getter are constexpr
    typedef ex<tup.template get<0>() > tt1;
    typedef exu<tup.template get<1>() > tt2;
    typedef exu<tup.template get<2>() > tt3;
}

