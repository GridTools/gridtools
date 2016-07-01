#pragma once

#include "stencil-composition/backend_metafunctions_fwd.hpp"

namespace gridtools {

    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyType >
    struct is_backend< backend< BackendId, GridId, StrategyType > > : boost::mpl::true_ {};

    template < typename T >
    struct backend_id;

    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyType >
    struct backend_id< backend< BackendId, GridId, StrategyType > >
        : enumtype::enum_type< enumtype::platform, BackendId > {};

} // gridtools
