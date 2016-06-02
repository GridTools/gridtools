#pragma once
#include <boost/mpl/equal.hpp>
#include "esf.hpp"

namespace gridtools {

    template < typename Esf1, typename Esf2 >
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT(
            (is_esf_descriptor< Esf1 >::value && is_esf_descriptor< Esf2 >::value), "Error: Internal Error");
        typedef static_bool< boost::is_same< typename Esf1::esf_function, typename Esf2::esf_function >::value &&
                             boost::mpl::equal< typename Esf1::args_t, typename Esf2::args_t >::value > type;
    };

    struct extract_esf_functor {
        template < typename Esf >
        struct apply {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Error");

            typedef typename Esf::esf_function type;
        };
    };

    template<typename Esf>
    struct esf_arg_list
    {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Error");
        typedef typename Esf::esf_function::arg_list type;
    };
}
