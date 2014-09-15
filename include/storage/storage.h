#pragma once
#include<common/defs.h>
#include"base_storage.h"

namespace gridtools {
    template < typename ValueType
             , typename Layout
             , bool IsTemporary = false
        >
    struct storage : public base_storage< enumtype::Host
                                         , ValueType
                                         , Layout
                                         , IsTemporary
                                         >
    {
    };
}//namespace gridtools
