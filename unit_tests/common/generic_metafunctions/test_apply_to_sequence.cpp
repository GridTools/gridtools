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
#include <boost/mpl/equal.hpp>
#include "defs.hpp"
#include "common/generic_metafunctions/apply_to_sequence.hpp"

using namespace gridtools;

template<int V>
struct metadata{
    static const int val=V;
};

template<typename Metadata>
struct extract_metadata
{
    typedef static_int<Metadata::val> type;
};

TEST(test_apply_to_sequence, test)
{
    typedef boost::mpl::vector3<metadata<3>, metadata<5>, metadata<8> > seq;

    typedef apply_to_sequence<seq, extract_metadata>::type res;
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<res, boost::mpl::vector<static_int<3>, static_int<5>, static_int<8> > >::value), "Wrong RESULT");

    ASSERT_TRUE(true);
}

