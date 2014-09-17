#pragma once
#include<common/defs.h>
#include"base_storage.h"

/**This class is a decorator of the storage, which implements some interfaces specific for the Cuda backend type*/
namespace gridtools {
    template < typename ValueType
             , typename Layout
             , bool IsTemporary = false
               >
    struct cuda_storage // : public base_storage< enumtype::Cuda
                        //                        , ValueType
                        //                        , Layout
                        //                        , IsTemporary
                        //                        >
    {
    };
} // namespace gridtools
