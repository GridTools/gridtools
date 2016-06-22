#include "gtest/gtest.h"

#include <common/explode_array.hpp>
#include <common/tuple.hpp>
#include <tools/verifier.hpp>

class PackChecker{

public:
    static bool check(int i1, int i2, int i3)
    {
        return i1==35 && i2==23 && i3==9;
    }

    template<typename ... UInt>
    static bool apply(UInt ... args)
    {
        return check(args...);
    }

};

class TuplePackChecker {

  public:
    static bool check(int i1, double i2, unsigned short i3) {
        return i1 == -35 && gridtools::compare_below_threshold(i2, 23.3, 1 - 8) && i3 == 9;
    }

    template < typename... UInt >
    static bool apply(UInt... args) {
        return check(args...);
    }
};

struct _impl_index{
    template<typename ... UInt>
    static bool apply(const PackChecker& me, UInt ... args){
        return me.check(args...);
    }
};

struct _impl_index_tuple {
    template < typename... UInt >
    static bool apply(const TuplePackChecker &me, UInt... args) {
        return me.check(args...);
    }
};

using namespace gridtools;

TEST(explode_array, test_explode_static) {
    array<int, 3> a{35,23,9};
    ASSERT_TRUE((explode<bool, PackChecker>(a)));
}

TEST(explode_array, test_explode_with_object) {
    array<int, 3> a{35,23,9};
    PackChecker checker;
    ASSERT_TRUE(( explode<int, _impl_index>(a, checker)));
}

TEST(explode_array, tuple) {
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    ASSERT_TRUE((explode< bool, TuplePackChecker >(a)));
}

TEST(explode_array, tuple_with_object) {
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    TuplePackChecker checker;
    ASSERT_TRUE((explode< bool, _impl_index_tuple >(a, checker)));
}
