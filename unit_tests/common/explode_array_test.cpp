#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "gtest/gtest.h"

#include <common/explode_array.hpp>


class PackChecker{

    static bool check(int i1, int i2, int i3)
    {
        return i1==35 && i2==23 && i3==9;
    }

public:
    template<typename ... UInt>
    static bool function(UInt ... args)
    {
        return check(args...);
    }

};
using namespace gridtools;

TEST(explode_array, test_explode) {
    array<int, 3> a(35,23,9);
    ASSERT_TRUE(explode<bool>(PackChecker::template function<int,int,int>, a));
}
