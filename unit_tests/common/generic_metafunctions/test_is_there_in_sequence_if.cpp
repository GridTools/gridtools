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
#include "common/generic_metafunctions/is_there_in_sequence_if.hpp"

using namespace gridtools;

TEST(is_there_in_sequence_if, is_there_in_sequence_if)
{
    typedef boost::mpl::vector<int, float, char> seq_t;

    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, char> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, int> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, long> >::value),"ERROR");

    ASSERT_TRUE(true);
}

