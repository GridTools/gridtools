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
#include "defs.hpp"
#include "common/generic_metafunctions/is_not_same.hpp"

using namespace gridtools;

TEST(is_not_same, test)
{
    GRIDTOOLS_STATIC_ASSERT((is_not_same<int, float>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_not_same<int, int>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_not_same<double, float>::value),"ERROR");

    ASSERT_TRUE(true);
}


