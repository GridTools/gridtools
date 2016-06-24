#include "gtest/gtest.h"
#include "defs.hpp"
#include "common/generic_metafunctions/accumulate.hpp"
#include "common/generic_metafunctions/logical_ops.hpp"
#include "common/array.hpp"

using namespace gridtools;

template < typename... Args >
GT_FUNCTION
static constexpr bool check_or(Args... args) {
    return accumulate(logical_or(), is_array< Args >::type::value...);
}

template < typename... Args >
GT_FUNCTION
static constexpr bool check_and(Args... args) {
    return accumulate(logical_and(), is_array< Args >::type::value...);
}

GT_FUNCTION
static bool test_accumulate_and() {
    GRIDTOOLS_STATIC_ASSERT((check_and(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((!check_and(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3}, 7)), "Error");

    return true;
}

GT_FUNCTION
static bool test_accumulate_or() {

    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3}, 7)), "Error");
    GRIDTOOLS_STATIC_ASSERT((!check_or(-2, 3, 7)), "Error");

    return true;
}
