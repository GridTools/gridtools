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
#include "common/is_aggregate.hpp"
#include "common/array.hpp"
#include "storage/wrap_pointer.hpp"

using namespace gridtools;

TEST(array, test_is_aggregate) {
    GRIDTOOLS_STATIC_ASSERT((is_aggregate< int >::value), "Error");

    typedef array< uint_t, 4 > array_t;
    GRIDTOOLS_STATIC_ASSERT((is_aggregate< array_t >::value), "Error");

    typedef wrap_pointer< double > ptr_t;
    GRIDTOOLS_STATIC_ASSERT((!is_aggregate< ptr_t >::value), "Error");

    ASSERT_TRUE(true);
}
