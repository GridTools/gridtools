#pragma once
#include<common/defs.h>
#include"base_storage.h"

namespace gridtools {
    template < typename BaseStorage >
    struct storage : public BaseStorage, clonable_to_gpu<storage<BaseStorage> >
    {
      typedef typename BaseStorage::basic_type basic_type;
      typedef storage<BaseStorage> original_storage;
      typedef clonable_to_gpu<storage<BaseStorage> > gpu_clone;

      __device__
      storage(storage const& other)
	: basic_type(other)
      {}
      
      // ~storage(){}
      // storage():BaseStorage(){}

    explicit storage(uint_t dim1, uint_t dim2, uint_t dim3,
		     typename BaseStorage::value_type init = BaseStorage::value_type(), std::string const& s = std::string("default name") ): BaseStorage(dim1, dim2, dim3, init, s) {
        }

    private : 
      storage();
    };

}//namespace gridtools
