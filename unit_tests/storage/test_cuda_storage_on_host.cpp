#include "gtest/gtest.h"
#include <iostream>
#define private public
#include <storage/storage.hpp>
#include <storage/meta_storage.hpp>

#ifdef _USE_GPU_ 
#include <storage/hybrid_pointer.hpp>
#else
#include <storage/wrap_pointer.hpp>
#endif

using namespace gridtools;

TEST(cuda_storage_on_host, test_storage) {
	using namespace gridtools;
	using namespace enumtype;

	typedef gridtools::layout_map< 0, 1, 2 > layout_t;
	typedef meta_storage<meta_storage_aligned< meta_storage_base<0, layout_t, false>, aligned< 32 >, halo< 0, 0, 0 > > > meta_data_t;
#ifdef _USE_GPU_
	typedef base_storage< hybrid_pointer<double>, meta_data_t, 1 > base_st;
#else
	typedef base_storage< wrap_pointer<double>, meta_data_t, 1> base_st;
#endif
	typedef storage< base_st > storage_t;
	meta_data_t meta_data(10, 10, 10);
	storage_t foo_field(meta_data);
	// fill storage
	int z=0;
	for(int i=0; i<10; ++i)
		for(int j=0; j<10; ++j)
			for(int k=0; k<10; ++k)
				foo_field(i, j, k) = z;
	// get storage ptr and on_host information
	ASSERT_TRUE(foo_field.m_on_host && "The storage should not be located on the device.");
	base_st* ptr1 = foo_field.m_storage.get_pointer_to_use();
#ifdef _USE_GPU_
	std::cout << "copy to device\n";
	foo_field.h2d_update();	
	base_st* ptr2 = foo_field.m_storage.get_pointer_to_use();
	ASSERT_FALSE(foo_field.m_on_host && "The storage should be located on the device.");
	ASSERT_TRUE(ptr1 != ptr2 && "Pointers to the storage must not be the same.");
	foo_field.d2h_update();	
	std::cout << "copy from device\n";
#endif
	base_st* ptr3 = foo_field.m_storage.get_pointer_to_use();
	ASSERT_TRUE(ptr1 == ptr3 && "Pointers to the storage must be the same.");	
	ASSERT_TRUE(foo_field.m_on_host && "The storage should not be located on the device.");
}
