#include <common/explode_array.hpp>
#include <common/tuple.hpp>
#include <tools/verifier.hpp>

class PackChecker {

  public:
    GT_FUNCTION
    static constexpr bool check(int i1, int i2, int i3) { return i1 == 35 && i2 == 23 && i3 == 9; }

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(UInt... args) {
        return check(args...);
    }
};

class TuplePackChecker {

  public:
    GT_FUNCTION
    static bool check(int i1, double i2, unsigned short i3) {
        return i1 == -35 && gridtools::compare_below_threshold(i2, 23.3, 1 - 8) && i3 == 9;
    }

    template < typename... UInt >
    GT_FUNCTION static bool apply(UInt... args) {
        return check(args...);
    }
};

class TuplePackCheckerInt {
  public:
    GT_FUNCTION
    static constexpr bool check(long i1, int i2, unsigned short i3) { return i1 == -353 && i2 == 55 && i3 == 9; }

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(UInt... args) {
        return check(args...);
    }
};

struct _impl_index {
    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const PackChecker &me, UInt... args) {
        return me.check(args...);
    }
};

struct _impl_index_tuple_int {
    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const TuplePackCheckerInt &me, UInt... args) {
        return me.check(args...);
    }
};

struct _impl_index_tuple {
    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const TuplePackChecker &me, UInt... args) {
        return me.check(args...);
    }
};

using namespace gridtools;

GT_FUNCTION
static bool test_explode_static() {
    constexpr array< int, 3 > a{35, 23, 9};
#ifndef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< bool, PackChecker >(a) >::value == true), "ERROR");
#endif
    return explode< bool, PackChecker >(a);
}

GT_FUNCTION
static bool test_explode_with_object() {
    constexpr array< int, 3 > a{35, 23, 9};
    constexpr PackChecker checker;
#ifndef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< int, _impl_index >(a, checker) >::value == true), "ERROR");
#endif
    return explode< int, _impl_index >(a, checker);
}

GT_FUNCTION
static bool test_explode_with_tuple() {
    bool result = true;

#ifndef __CUDACC__
    // constexpr check
    constexpr tuple< long, int, unsigned short > a_c(-353, 55, 9);
    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< bool, TuplePackCheckerInt >(a_c) >::value == true), "ERROR");
    result = result && explode< bool, TuplePackCheckerInt >(a_c);
#endif
    // with a double constexpr check is not possible
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    result = result && explode< bool, TuplePackChecker >(a);
    return result;
}

GT_FUNCTION
static bool test_explode_with_tuple_with_object() {
    bool result = true;
#ifndef __CUDACC__
    // constexpr check
    constexpr tuple< long, int, unsigned short > a_c(-353, 55, 9);
    constexpr TuplePackCheckerInt checker_c;
    GRIDTOOLS_STATIC_ASSERT(
        (static_bool< explode< bool, _impl_index_tuple_int >(a_c, checker_c) >::value == true), "ERROR");
    result = result && explode< bool, _impl_index_tuple_int >(a_c, checker_c);
#endif
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    TuplePackChecker checker;
    result = result && explode< bool, _impl_index_tuple >(a, checker);
}
