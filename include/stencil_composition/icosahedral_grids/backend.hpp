#pragma once

#include "storage/storage.hpp"
#include "storage/meta_storage.hpp"
#include "location_type.hpp"
#include "stencil_composition/backend_base.hpp"
#include "storage/wrap_pointer.hpp"
#include "icosahedral_grid_traits.hpp"

namespace gridtools {

    /**
       The backend is, as usual, declaring what the storage types are
     */
    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend< BackendId, enumtype::icosahedral, StrategyType >
        : public backend_base< BackendId, enumtype::icosahedral, StrategyType > {
      public:
        // template <typename LocationType, typename X, typename LayoutMap>
        // struct _storage_type;

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<0, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<0, NColors> >;
        // };

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<1, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<1, NColors> >;
        // };

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<2, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<2, NColors> >;
        // };

        typedef backend_base< BackendId, enumtype::icosahedral, StrategyType > base_t;

        using typename base_t::backend_traits_t;
        using typename base_t::strategy_traits_t;

        // TODO storage and meta_storage have to be moved to backend_traits_from_id, that has to be templated with grid
        template < typename LocationType >
        using storage_info_t = typename base_t::template storage_info< LocationType::value,
            typename icgrid::grid_traits_arch< base_t::s_backend_id >::layout_map_t >;

        template < typename LocationType, typename ValueType >
        using storage_t = typename base_t::template storage_type< ValueType, storage_info_t< LocationType > >::type;
    };
} // namespace gridtools
