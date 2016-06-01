#pragma once

namespace gridtools {
    template < typename T >
    struct is_actual_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_temporary_storage : boost::mpl::false_ {};
} // namespace gridtools
