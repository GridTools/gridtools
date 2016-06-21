#pragma once
#include "common/defs.hpp"
#include "common/tuple.hpp"


#ifdef CXX11_ENABLED
GT_FUNCTION
void test_tuple_elements(bool *result)
{
    using namespace gridtools;

    *result = true;
    constexpr tuple<int_t, short_t, uint_t> tup(-3,4,10);

    GRIDTOOLS_STATIC_ASSERT((static_int<tup.get<0>()>::value == -3), "ERROR");

#if defined(CXX11_ENABLED) && !defined(__CUDACC__)

    // CUDA does not think the following are constexprable :(
    GRIDTOOLS_STATIC_ASSERT((static_int<tup.n_dimensions>::value == 3), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_int<tup.get<1>()>::value == 4), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_int<tup.get<2>()>::value == 10), "ERROR");
#endif

    *result &= ((tup.get<0>() == -3));
    *result &= ((tup.get<1>() == 4));
    *result &= ((tup.get<2>() == 10));

}

#endif
