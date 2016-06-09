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
