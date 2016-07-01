#include <common/explode_array.hpp>
#include <common/tuple.hpp>
#include <tools/verifier.hpp>

#if !defined(__CUDACC__) || (CUDA_VERSION_MAJOR >= 7 && CUDA_VERSION_MINOR>=5)
class PackChecker {

  public:
    constexpr PackChecker() {}

    GT_FUNCTION
    static constexpr bool check(int i1, int i2, int i3) { return i1 == 35 && i2 == 23 && i3 == 9; }

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(UInt... args) {
        return check(args...);
    }
};

class TuplePackChecker {

  public:
    constexpr TuplePackChecker() {}
    GT_FUNCTION
    static bool check(int i1, double i2, unsigned short i3) {
        return ((i1 == -35) && gridtools::compare_below_threshold(i2, 23.3, 1e-6) && (i3 == 9));
    }

    template < typename... UInt >
    GT_FUNCTION static bool apply(UInt... args) {
        return check(args...);
    }
};

class TuplePackCheckerInt {
  public:
    constexpr TuplePackCheckerInt() {}

    GT_FUNCTION
    static constexpr bool check(long i1, int i2, unsigned short i3) { return i1 == -353 && i2 == 55 && i3 == 9; }

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(UInt... args) {
        return check(args...);
    }
};

struct _impl_index {
    constexpr _impl_index() {}

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const PackChecker &me, UInt... args) {
        return me.check(args...);
    }
};

struct _impl_index_tuple_int {
    constexpr _impl_index_tuple_int() {}

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const TuplePackCheckerInt &me, UInt... args) {
        return me.check(args...);
    }
};

struct _impl_index_tuple {
    constexpr _impl_index_tuple() {}

    template < typename... UInt >
    GT_FUNCTION static constexpr bool apply(const TuplePackChecker &me, UInt... args) {
        return me.check(args...);
    }
};
#endif

using namespace gridtools;

GT_FUNCTION
static bool test_explode_static() {
#if !defined(__CUDACC__) || (CUDA_VERSION_MAJOR >= 7 && CUDA_VERSION_MINOR>=5)

    constexpr array< int, 3 > a{35, 23, 9};

    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< bool, PackChecker >(a) >::value == true), "ERROR");
    return explode< bool, PackChecker >(a);
#else
    return true;
#endif
}

GT_FUNCTION
static bool test_explode_with_object() {
#if !defined(__CUDACC__) || (CUDA_VERSION_MAJOR >= 7 && CUDA_VERSION_MINOR>=5)
    constexpr array< int, 3 > a{35, 23, 9};
    constexpr PackChecker checker;

    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< int, _impl_index >(a, checker) >::value == true), "ERROR");
    return explode< int, _impl_index >(a, checker);
#else
    return true;
#endif
}

GT_FUNCTION
static bool test_explode_with_tuple() {
    bool result = true;

#if !defined(__CUDACC__) || (CUDA_VERSION_MAJOR >= 7 && CUDA_VERSION_MINOR>=5)
    // constexpr check
    constexpr tuple< long, int, unsigned short > a_c(-353, 55, 9);
#if (CUDA_VERSION_MAJOR > 7)
    GRIDTOOLS_STATIC_ASSERT((static_bool< explode< bool, TuplePackCheckerInt >(a_c) >::value == true), "ERROR");
#endif
    result = result && explode< bool, TuplePackCheckerInt >(a_c);
    // with a double constexpr check is not possible
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    result = result && explode< bool, TuplePackChecker >(a);
#endif
    return result;
}

GT_FUNCTION
static bool test_explode_with_tuple_with_object() {
    bool result = true;
#if !defined(__CUDACC__) || (CUDA_VERSION_MAJOR >= 7 && CUDA_VERSION_MINOR>=5)
    // constexpr check
    constexpr tuple< long, int, unsigned short > a_c(-353, 55, 9);
    constexpr TuplePackCheckerInt checker_c;
#if (CUDA_VERSION_MAJOR > 7)
    GRIDTOOLS_STATIC_ASSERT(
        (static_bool< explode< bool, _impl_index_tuple_int >(a_c, checker_c) >::value == true), "ERROR");
#endif
    result = result && explode< bool, _impl_index_tuple_int >(a_c, checker_c);
    tuple< int, float, unsigned short > a(-35, 23.3, 9);
    TuplePackChecker checker;
    result = result && explode< bool, _impl_index_tuple >(a, checker);
#endif
    return result;
}
