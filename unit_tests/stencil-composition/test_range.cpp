/*
 * test_backend.cpp
 *
 *  Created on: Mar 11, 2015
 *      Author: carlosos
 */

#include <gridtools.h>
#include <boost/mpl/equal.hpp>

#include "gtest/gtest.h"

using namespace gridtools;

TEST(test_range, union_ranges) {

    using namespace gridtools;

    {
    typedef range<-2,2,-4,5> range1_t;
    typedef range<-1,3,-7,4> range2_t;
    typedef range<-2,3,-7,5> exp_range_t;
    BOOST_STATIC_ASSERT((
        boost::is_same<
            union_ranges<range1_t, range2_t>::type,
            exp_range_t
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            union_ranges<range2_t, range1_t>::type,
            exp_range_t
        >::value
    ));
    }

    {
    typedef range<-2,2,-4,5> range1_t;
    typedef range<-3,1,-3,7> range2_t;
    typedef range<-3,2,-4,7> exp_range_t;
    BOOST_STATIC_ASSERT((
        boost::is_same<
            union_ranges<range1_t, range2_t>::type,
            exp_range_t
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            union_ranges<range2_t, range1_t>::type,
            exp_range_t
        >::value
    ));
    }

    EXPECT_TRUE(true);
}

