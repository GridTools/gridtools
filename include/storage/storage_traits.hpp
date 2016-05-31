#pragma once
#include "../common/defs.hpp"
#include "../common/layout_map.hpp"
#include "halo.hpp"

#include "storage_traits_cuda.hpp"
#include "storage_traits_host.hpp"

namespace gridtools {

    template < enumtype::platform T >
    struct storage_traits_t {

        typedef gridtools::storage_traits_from_id<T> storage_traits_aux;

        // convenience structs that unwrap the inner storage and meta_storage type
#ifdef CXX11_ENABLED
        template < typename ValueType, typename MetaData >
        using storage_type = typename storage_traits_aux::template storage_t<ValueType, MetaData, false>::type;

        template < ushort_t Index, typename Layout, bool Temp, typename Halo = halo<0,0,0>, typename Alignment = typename storage_traits_aux::default_alignment::type>
        using meta_storage_type = typename storage_traits_aux::template meta_storage_t< static_uint< Index >, Layout, Temp, Halo, Alignment>::type;
#else // CXX11_ENABLED
        template < typename ValueType, typename MetaData >
        struct storage_type {
            typedef typename storage_traits_aux::template storage_t<ValueType, MetaData, false>::type type;
        };

        template < ushort_t Index, typename Layout, bool Temp, typename Halo = halo<0,0,0>, typename Alignment = typename storage_traits_aux::default_alignment::type>
        struct meta_storage_type {
            typedef typename storage_traits_aux::template meta_storage_t< static_uint< Index >, Layout, Temp, Halo, Alignment>::type type;
        };
#endif
    };

};
