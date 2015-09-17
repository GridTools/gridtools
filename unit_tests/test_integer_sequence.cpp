#include "gtest/gtest.h"
#include "../include/common/generic_metafunctions/gt_integer_sequence.hpp"
#include "common/array.hpp"

// using namespace gridtools;

template<gridtools::uint_t Idx>
struct get_component{

GT_FUNCTION
constexpr get_component(){}

template<typename OtherArray>
GT_FUNCTION
constexpr int& apply(OtherArray const& other_){
return other_[Idx];
}
};
template< typename T>
struct apply_gt_integer_sequence{};

TEST(integer_sequence, fill_array) {

    constexpr gridtools::array<int, 4> in{0,1,2,3};


    using seq = gridtools::apply_gt_integer_sequence<gridtools::make_gt_integer_sequence<int, 4> >;
    //calling the array constexpr copy constructor
    constexpr array<int, 4> out( seq::template apply<array<T, 4>, get_component>(0,1,2,3) );

    GRIDTOOLS_STATIC_ASSERT( out[0]==0 && out[1]==1 && out[2]==2 && out[3]==3, "Error in tet_integer_sequence");
}
