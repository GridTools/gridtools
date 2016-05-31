#pragma once
#include "../common/defs.hpp"
#include "storage.hpp"
#include "base_storage.hpp"
#include "meta_storage.hpp"
#include "meta_storage_aligned.hpp"
#include "meta_storage_base.hpp"

namespace gridtools {
    template < enumtype::platform T >
    struct storage_traits_from_id;

    /**Traits struct, containing the types which are specific for the host backend*/
    template <>
    struct storage_traits_from_id< enumtype::Host > {

        /**
           @brief pointer type associated to the host backend
         */
        template < typename T >
        struct pointer {
            typedef wrap_pointer< T, true > type;
        };

        /**
           @brief storage type associated to the host backend
         */
        template < typename ValueType, typename MetaData, bool Temp, short_t FieldDim = 1 >
        struct storage_t {
            GRIDTOOLS_STATIC_ASSERT((is_meta_storage< MetaData >::value), "wrong type for the storage_info");
            typedef storage< base_storage< typename pointer< ValueType >::type, MetaData, FieldDim > > type;
        };

        struct default_alignment {
            typedef aligned< 0 > type;
        };

        /**
           @brief storage info type associated to the host backend

           the storage info type is meta_storage_base, which is not clonable to GPU.
         */
        template < typename IndexType, typename Layout, bool Temp, typename Halo, typename Alignment >
        struct meta_storage_t {
            GRIDTOOLS_STATIC_ASSERT((is_layout_map< Layout >::value), "wrong type for the storage_info");
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_aligned< Alignment >::type::value, "wrong type");

            typedef meta_storage<
                meta_storage_aligned< meta_storage_base< IndexType::value, Layout, Temp >, Alignment, Halo > > type;
        };

    };
}
