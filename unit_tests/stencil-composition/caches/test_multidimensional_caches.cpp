
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

namespace test_multidimensional_caches{
using namespace gridtools;

    int test(){
        typedef layout_map<0,1,2,3,4,5> layout_t;
        typedef pointer<base_storage<wrap_pointer<double>, meta_storage_base<0, layout_t, false>, 4 > > storage_t;
        typedef cache_storage<double, block_size<8,3,4,5,6>, extent<-1,1 ,-2,2 ,0,2 ,0,0, -1,0>, storage_t > cache_storage_t;
        typedef accessor<0,enumtype::in,extent<>,6> acc_t;

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(0)==10, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(1)==7, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(2)==6, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(3)==1, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(4)==7, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::value().dims(5)==4, "error");
        // cache_storage_t::meta_storage_t::fuck();

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t{0,0,0,0,0,0})==0, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,0,1))==1, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,0,2))==2, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,0,3))==3, "error");

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,1,0))==4, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,2,0))==8, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,3,0))==12, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,4,0))==16, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,5,0))==20, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,0,6,0))==24, "error");

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,0,1,0,0))==28, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,1,0,0,0))==28, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,2,0,0,0))==56, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,3,0,0,0))==84, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,4,0,0,0))==112, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,0,5,0,0,0))==140, "error");

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,1,0,0,0,0))==168, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,2,0,0,0,0))==336, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,3,0,0,0,0))==504, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,4,0,0,0,0))==672, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,5,0,0,0,0))==840, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(0,6,0,0,0,0))==1008, "error");

        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(1,0,0,0,0,0))==1176, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(2,0,0,0,0,0))==2352, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(3,0,0,0,0,0))==3528, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(4,0,0,0,0,0))==4704, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(5,0,0,0,0,0))==5880, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(6,0,0,0,0,0))==7056, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(7,0,0,0,0,0))==8232, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(8,0,0,0,0,0))==9408, "error");
        GRIDTOOLS_STATIC_ASSERT(cache_storage_t::meta_storage_t::index(acc_t(9,0,0,0,0,0))==10584, "error");


        typedef cache_storage<double, block_size<8,3,4,5,6>, extent<-1,1 ,0,0 ,0,0 ,0,0, 0,0>, storage_t > cache_storage2_t;
        typedef accessor<0,enumtype::in,extent<>,3> acc2_t;
        cache_storage2_t cache_;

        return 0;
    }
}//namespace test_multidimensional_caches

TEST(define_caches, test_sequence_caches)
{
    test_multidimensional_caches::test();
}
