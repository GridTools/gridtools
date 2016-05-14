#include "gtest/gtest.h"
#include "defs.hpp"
#include "common/generic_metafunctions/accumulate.hpp"
#include "common/generic_metafunctions/logical_ops.hpp"
#include "common/array.hpp"

using namespace gridtools;

template<typename ... Args>
bool check_or(Args... args)
{
    return accumulate(logical_or(), is_array< Args>::type::value...);
}

template<typename ... Args>
bool check_and(Args... args)
{
    return accumulate(logical_and(), is_array< Args>::type::value...);
}


TEST(accumulate, test_and) {

    ASSERT_TRUE(check_and(array<uint_t, 4>{3,4,5,6}, array<int_t, 2>{-2,3}) );
    ASSERT_TRUE(!check_and(array<uint_t, 4>{3,4,5,6}, array<int_t, 2>{-2,3}, 7) );

}

TEST(accumulate, test_or) {

    ASSERT_TRUE(check_or(array<uint_t, 4>{3,4,5,6}, array<int_t, 2>{-2,3}) );
    ASSERT_TRUE(check_or(array<uint_t, 4>{3,4,5,6}, array<int_t, 2>{-2,3}, 7) );
    ASSERT_TRUE(!check_or(-2,3, 7) );
}
