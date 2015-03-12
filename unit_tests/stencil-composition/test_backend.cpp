/*
 * test_backend.cpp
 *
 *  Created on: Mar 11, 2015
 *      Author: carlosos
 */

#include <gridtools.h>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif

using namespace gridtools;

TEST(test_backend, merge_range_temporary_maps) {

    using namespace gridtools;

    typedef gridtools::layout_map<0,1,2> layout_ijk;
    typedef gridtools::backend<gridtools::enumtype::Host, enumtype::Naive > test_backend_t;
    typedef test_backend_t::storage_type<gridtools::float_type, layout_ijk >::type storage_type;

    typedef arg<0, storage_type> p_field1;
    typedef arg<1, storage_type> p_field2;
    typedef arg<2, storage_type> p_field3;
    typedef arg<3, storage_type> p_field4;
    typedef arg<4, storage_type> p_field5;

    typedef boost::mpl::map2<
        boost::mpl::pair<p_field1, range<-1,1,-1,1> >,
        boost::mpl::pair<p_field2, range<-1,1,-1,1> >
    > map1_t;

    typedef boost::mpl::map3<
        boost::mpl::pair<p_field2, range<0,3,-2,1> >,
        boost::mpl::pair<p_field4, range<-1,3,-2,1> >,
        boost::mpl::pair<p_field1, range<-9,0,-2,0> >
    > map2_t;

    {
    typedef backend<enumtype::Host, enumtype::Naive>::merge_range_temporary_maps<map1_t, map2_t>::type merged_map_t;

    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::mpl::at<merged_map_t, p_field1>::type,
            range<-9,1,-2,1>
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::mpl::at<merged_map_t, p_field2>::type,
            range<-1,3,-2,1>
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::mpl::at<merged_map_t, p_field4>::type,
            range<-1,3,-2,1>
        >::value
    ));
    }

    {
        typedef backend<enumtype::Host, enumtype::Naive>::merge_range_temporary_maps<map2_t, map1_t>::type merged_map_t;

        BOOST_STATIC_ASSERT((
            boost::is_same<
                boost::mpl::at<merged_map_t, p_field1>::type,
                range<-9,1,-2,1>
            >::value
        ));
        BOOST_STATIC_ASSERT((
            boost::is_same<
                boost::mpl::at<merged_map_t, p_field2>::type,
                range<-1,3,-2,1>
            >::value
        ));
        BOOST_STATIC_ASSERT((
            boost::is_same<
                boost::mpl::at<merged_map_t, p_field4>::type,
                range<-1,3,-2,1>
            >::value
        ));

    }

    EXPECT_TRUE(true);
}

