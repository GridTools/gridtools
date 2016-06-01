#pragma once
#include <stencil-composition/stencil-composition.hpp>

GT_FUNCTION
void test_offset_tuple(bool *result) {
    using namespace gridtools;
    *result = true;
#if defined(NDEBUG) && defined(CXX11_ENABLED)
    {
    constexpr array<int_t, 4> pos{2,5,8,-6};
    constexpr offset_tuple<4,4> offsets(0, pos);

    GRIDTOOLS_STATIC_ASSERT((static_int<offsets.get<0>() >::value == -6), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int<offsets.get<1>() >::value == 8), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int<offsets.get<2>() >::value == 5), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int<offsets.get<3>() >::value == 2), "Error");
    }
#endif
    {
    array<int_t, 4> pos{2,5,8,-6};
    offset_tuple<4,4> offsets(0, pos);

    *result &= ((offsets.get<0>() == -6));
    *result &= ((offsets.get<1>() == 8));
    *result &= ((offsets.get<2>() == 5));
    *result &= ((offsets.get<3>() == 2));
    }
}
