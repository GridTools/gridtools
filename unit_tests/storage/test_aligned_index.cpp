#include "gtest/gtest.h"

#include <iostream>
#include <storage/storage-facility.hpp>

using namespace gridtools;
using namespace enumtype;

typedef layout_map<1,0> layout;

template <typename T>
void print_me(T& storage, int splitter) {
    auto size = storage.fields_view()->get_size();
    std::cout << "size: " << size << "\n" << std::endl;
    for(int i=0; i<size; ++i) {
        std::cout << (*storage.fields_view())[i] << " ";
        if((i+1)%splitter == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

TEST(aligned_index_test, unaligned_access) {
    typedef meta_storage< meta_storage_base<0, layout, false> > meta_storage_t;
    typedef storage< base_storage< wrap_pointer< double, true >, meta_storage_t, 1 > > storage_t;
    meta_storage_t meta(3,3);
    storage_t data(meta, 0.0, "data");
    int bla = 0;
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            data(i,j) = bla++;
        }
    }
    print_me(data, 3);
}


TEST(aligned_index_test, aligned_access) {
    typedef meta_storage< meta_storage_aligned< meta_storage_base<0, layout, false>, aligned<8>, halo<2,0> > > meta_storage_t;
    typedef storage< base_storage< wrap_pointer< double, true >, meta_storage_t, 1 > > storage_t;
    meta_storage_t meta(3,3);
    storage_t data(meta, 0.0, "data");
    //test the storage
    int bla = 0;    
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            data(i,j) = bla++;
        }
    }
    print_me(data, 16);    
}

