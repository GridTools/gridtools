#pragma once

#include "stencil-composition/backend_metafunctions_fwd.hpp"

namespace gridtools {

    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct is_backend< backend< BackendId, StrategyType > > : boost::mpl::true_ {};

    template < typename T >
    struct backend_id;

    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend_id< backend< BackendId, StrategyType > > : enumtype::enum_type< enumtype::platform, BackendId > {};

} // gridtools
