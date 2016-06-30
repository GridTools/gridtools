#pragma once

namespace gridtools {
    /**
     * type trait that removes __restrict__ qualifier from a type
     */
    template < typename T >
    struct remove_restrict_reference {
        typedef T type;
    };

    template < typename T >
    struct remove_restrict_reference< T & RESTRICT > {
        typedef T type;
    };
}
