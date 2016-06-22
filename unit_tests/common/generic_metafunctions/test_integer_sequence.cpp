#include "gtest/gtest.h"
#include "common/generic_metafunctions/gt_integer_sequence.hpp"
#include "common/array.hpp"
#include "common/generic_metafunctions/variadic_typedef.hpp"

#ifdef NDEBUG

template < int Idx >
struct get_component {

    GT_FUNCTION
    constexpr get_component() {}

    template < typename... Ints >
    GT_FUNCTION constexpr static int apply(Ints... args_) {
        return std::get< Idx >(std::make_tuple(args_...));
    }
};
using namespace gridtools;

TEST(integer_sequence, fill_array) {

    using seq = gridtools::apply_gt_integer_sequence< typename gridtools::make_gt_integer_sequence< int, 4 >::type >;

    // calling the array constexpr copy constructor
    constexpr gridtools::array< int, 4 > out(
        seq::template apply< gridtools::array< int, 4 >, get_component >(0, 1, 2, 3));

    // verifying that the information is actually compile-time known and that it's correct
    GRIDTOOLS_STATIC_ASSERT(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3, "Error in test_integer_sequence");
}

template < int Idx >
struct transform {

    GT_FUNCTION
    constexpr transform() {}

    template < typename... Args >
    GT_FUNCTION constexpr static int apply(Args... args) {
        return get_from_variadic_pack< Idx >::apply(args...) - Idx;
    }
};

struct lambda {
    constexpr int operator()(const int i, const int j, const int k, const int l, const int add) {
        return add * (i + j + k + l);
    }
};

TEST(integer_sequence, apply_lambda) {

    using seq = gridtools::apply_gt_integer_sequence< typename gridtools::make_gt_integer_sequence< int, 4 >::type >;

    constexpr auto gather = lambda();

    constexpr int result = seq::template apply_lambda< int, decltype(gather), transform >(gather, 17, 4, 6, 34, 5);

    GRIDTOOLS_STATIC_ASSERT((static_int< result >::value == 731), "ERROR");

    ASSERT_TRUE(true);
}

#endif
