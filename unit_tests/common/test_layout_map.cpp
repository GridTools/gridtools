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
#include "test_layout_map.hpp"

using namespace gridtools;

TEST(layout_map, accessors) {
    bool result = true;
    test_layout_accessors(&result);
    ASSERT_TRUE(result);
}

TEST(layout_map, find_val) {
    bool result = true;
    test_layout_find_val(&result);

    ASSERT_TRUE(&result);
}
