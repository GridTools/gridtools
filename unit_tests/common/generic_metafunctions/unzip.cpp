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
#include "common/generic_metafunctions/unzip.hpp"
#include <tuple>

using namespace gridtools;

TEST(unzip, do_unzip) {

    typedef std::tuple<int, float, double, char, bool, short> list_t;
    //verifying that the information is actually compile-time known and that it's correct
    static_assert( std::is_same<std::tuple<int, double, bool>, typename unzip<list_t>::first >::value, "error on first argument" );
    static_assert( std::is_same<std::tuple<float, char, short>, typename unzip<list_t>::second >::value, "error on second argument" );
}
