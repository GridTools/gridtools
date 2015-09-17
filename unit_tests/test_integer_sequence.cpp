#include "gtest/gtest.h"
#include "../include/common/generic_metafunctions/gt_integer_sequence.hpp"
#include "common/array.hpp"

template<int Idx>
struct get_component{

    GT_FUNCTION
    constexpr get_component(){}

    template<typename ... Ints>
    GT_FUNCTION
    constexpr static int apply(Ints ... args_){
        return std::get<Idx>(std::make_tuple (args_...));
    }
};

TEST(integer_sequence, fill_array) {

    //calling the array constexpr constructor
    constexpr gridtools::array<int, 4> in{0,1,2,3};

    using seq = gridtools::apply_gt_integer_sequence<gridtools::make_gt_integer_sequence<int, 4>::type >;

    //calling the array constexpr copy constructor
    constexpr gridtools::array<int, 4> out( seq::template apply<gridtools::array<int, 4>, get_component>(0,1,2,3) );

    //verifying that the information is actually compile-time known and that it's correct
    GRIDTOOLS_STATIC_ASSERT( out[0]==0 && out[1]==1 && out[2]==2 && out[3]==3, "Error in test_integer_sequence");
}
