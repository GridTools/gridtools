#pragma once

namespace gridtools {
    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend;

    // traits for backend
    template < typename T >
    struct is_backend : boost::mpl::false_ {};

} // gridtools
