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

#include "explode_array_test.hpp"

TEST(explode_array, test_explode_static) { ASSERT_TRUE(test_explode_static()); }

TEST(explode_array, test_explode_with_object) { ASSERT_TRUE(test_explode_with_object()); }

TEST(explode_array, tuple) { ASSERT_TRUE((test_explode_with_tuple())); }

TEST(explode_array, tuple_with_object) { ASSERT_TRUE((test_explode_with_tuple_with_object())); }
