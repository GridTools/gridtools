#pragma once
#include "../common/defs.hpp"
#include "../common/layout_map.hpp"
#include "halo.hpp"

#ifdef __CUDACC__
#include "storage_traits_cuda.hpp"
#else
#include "storage_traits_host.hpp"
#endif

namespace gridtools {
    template < enumtype::platform T >
    struct storage_traits_from_id;

    struct storage_traits_t {
        // get the correct storage_traits_from_id either for cuda or for host
#ifdef __CUDACC__
        typedef gridtools::storage_traits_from_id<enumtype::Cuda> storage_traits_aux;
#else
        typedef gridtools::storage_traits_from_id<enumtype::Host> storage_traits_aux;
#endif
        // convenience structs that unwrap the inner storage and meta_storage type
        template < typename ValueType, typename MetaData >
        struct storage_type : public storage_traits_aux::template storage_t<ValueType, MetaData, false>::type {
            typedef typename storage_traits_aux::template storage_t<ValueType, MetaData, false>::type super;
            using super::super;
        };
        template < ushort_t Index, typename Layout, bool Temp, typename Halo = halo<0,0,0>, typename Alignment = typename storage_traits_aux::default_alignment::type>
        struct meta_storage_type : public storage_traits_aux::template meta_storage_t< static_uint< Index >, Layout, Temp, Halo, Alignment>::type {
            typedef typename storage_traits_aux::template meta_storage_t< static_uint< Index >, Layout, Temp, Halo, Alignment>::type super;
            using super::super;
        };
    };

    template < ushort_t Index, typename Layout, bool Temp, typename Halo, typename Alignment>
    struct is_meta_storage< storage_traits_t::meta_storage_type<Index, Layout, Temp, Halo, Alignment> > : boost::mpl::true_ {};
};
