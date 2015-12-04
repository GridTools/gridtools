#ifdef CXX11_ENABLED
#include "gtest/gtest.h"
#include "storage/meta_storage.hpp"
#include "storage/storage.hpp"
#include "storage/wrap_pointer.hpp"

using namespace gridtools;

TEST(storage, test_data_field)
{
    typedef base_storage<wrap_pointer<int>, backend<Host, Naive >::storage_info<0, layout_map<0,1> > > storage_t;
    backend<Host, Naive >::storage_info<0, layout_map<0,1> > meta_(1,1);

    field<storage_t, 3,2,4>::type datafield(meta_, 0, "data");

    datafield.get_value<1,0>(0,0)=1;
    datafield.get_value<2,0>(0,0)=2;

    datafield.get_value<0,1>(0,0)=10;
    datafield.get_value<1,1>(0,0)=11;

    datafield.get_value<0,2>(0,0)=100;
    datafield.get_value<1,2>(0,0)=101;
    datafield.get_value<2,2>(0,0)=102;
    datafield.get_value<3,2>(0,0)=103;

    /*swaps the first and last snapshots of the first dimension*/
    swap<0,0>::with<2,0>::apply(datafield);
    assert((datafield.get_value<0,0>(0,0)==2 && datafield.get_value<2,0>(0,0)==0));

    std::cout<<"STORAGE VALUES BEFORE: "<<datafield.get_value<0,2>(0,0)
             <<" "<< datafield.get_value<1,2>(0,0)
             <<" "<< datafield.get_value<2,2>(0,0)
             <<" "<< datafield.get_value<3,2>(0,0)<<std::endl;

    /*advance the third dimension*/
    advance<2>::apply(datafield);
    assert((datafield.get_value<0,2>(0,0)==103
            && datafield.get_value<1,2>(0,0)==100
            && datafield.get_value<2,2>(0,0)==101
            && datafield.get_value<3,2>(0,0)==102
               ));

    ASSERT_TRUE(true);
}
#endif
