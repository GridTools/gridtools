#pragma once

namespace gridtools {

    template < typename LocalDomain, typename Accessor >
    struct has_arg {
        typedef typename is_arg< typename get_arg_accessor< LocalDomain, Accessor >::type >::type type;
    };

} // namespace gridtools
