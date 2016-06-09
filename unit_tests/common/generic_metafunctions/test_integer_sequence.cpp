/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "gtest/gtest.h"
#include "common/generic_metafunctions/gt_integer_sequence.hpp"
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
using namespace gridtools;

TEST(integer_sequence, fill_array) {

    using seq = gridtools::apply_gt_integer_sequence<
        typename gridtools::make_gt_integer_sequence<int, 4>::type
    >;

    //calling the array constexpr copy constructor
    constexpr gridtools::array<int, 4> out( seq::template apply<gridtools::array<int, 4>, get_component>(0,1,2,3) );

    //verifying that the information is actually compile-time known and that it's correct
    GRIDTOOLS_STATIC_ASSERT( out[0]==0 && out[1]==1 && out[2]==2 && out[3]==3, "Error in test_integer_sequence");
}
