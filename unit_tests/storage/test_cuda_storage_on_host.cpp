#include "gtest/gtest.h"

//i know that the following directive is super ugly, 
//but i need to check the private member fields of 
//the storage.
#define private public 
#include <storage/storage-t.hpp>

#ifdef _USE_GPU_ 
#include <storage/hybrid_pointer.hpp>
#else
#include <storage/wrap_pointer.hpp>
#endif

using namespace gridtools;

TEST(cuda_storage_on_host, test_storage_types) {
	typedef layout_map<0,1,2> layout;
	typedef storage_traits_t::meta_storage_type<0, layout, false> meta_data_t;
	typedef storage_traits_t::storage_type<float, meta_data_t> storage_t;
	meta_data_t meta_obj(10,10,10);
	storage_t st_obj(meta_obj, "in");
#ifdef _USE_GPU_
	GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename meta_data_t::super, meta_storage< meta_storage_aligned< meta_storage_base< 0, layout, false >, aligned<32>, halo<0,0,0> > > >::value), "type is wrong");
	GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename storage_t::super, storage< base_storage< hybrid_pointer< float >, meta_data_t, 1 > > >::value), "type is wrong");
#else
	GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename meta_data_t::super, meta_storage< meta_storage_aligned< meta_storage_base< 0, layout, false >, aligned<0>, halo<0,0,0> > > >::value), "type is wrong");
	GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename storage_t::super, storage< base_storage< wrap_pointer< float >, meta_data_t, 1 > > >::value), "type is wrong");
#endif
}

TEST(cuda_storage_on_host, test_storage) {
	using namespace gridtools;
	using namespace enumtype;
	// some typedefs to create a storage.
	// either a host backend storage or a 
	// cuda backend storage.
	typedef gridtools::layout_map< 0, 1, 2 > layout_t;
	typedef meta_storage<meta_storage_aligned< meta_storage_base<0, layout_t, false>, aligned< 32 >, halo< 0, 0, 0 > > > meta_data_t;
#ifdef _USE_GPU_
	typedef base_storage< hybrid_pointer<double>, meta_data_t, 1 > base_st;
#else
	typedef base_storage< wrap_pointer<double>, meta_data_t, 1> base_st;
#endif
	typedef storage< base_st > storage_t;
	// initializer the meta_data and the storage
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
	// copy the field to the gpu and check
	// for correct behaviour.
	foo_field.h2d_update();	
	base_st* ptr2 = foo_field.m_storage.get_pointer_to_use();
	ASSERT_FALSE(foo_field.m_on_host && "The storage should be located on the device.");
	ASSERT_TRUE(ptr1 != ptr2 && "Pointers to the storage must not be the same.");
	// copy the field back from the gpu
	foo_field.d2h_update();	
#endif
	// check if the pointers are right and the
	// field is on the host again.
	base_st* ptr3 = foo_field.m_storage.get_pointer_to_use();
	ASSERT_TRUE(ptr1 == ptr3 && "Pointers to the storage must be the same.");	
	ASSERT_TRUE(foo_field.m_on_host && "The storage should not be located on the device.");
}
