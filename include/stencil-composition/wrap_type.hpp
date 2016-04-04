#pragma once

namespace gridtools {
    namespace _impl {
        /**@brief wrap type to simplify specialization based on mpl::vectors */
        template < typename MplArray >
        struct wrap_type {
            typedef MplArray type;
        };

        /**
     * @brief compile-time boolean operator returning true if the template argument is a wrap_type
     * */
        template < typename T >
        struct is_wrap_type : boost::false_type {};

        template < typename T >
        struct is_wrap_type< wrap_type< T > > : boost::true_type {};
    }
}
