#pragma once

namespace gridtools {
    template < typename T >
    struct remove_restrict_reference {
        typedef T type;
    };

    template < typename T >
    struct remove_restrict_reference< T & RESTRICT > {
        typedef T type;
    };
}
