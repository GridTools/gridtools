#pragma once
#include <boost/type_traits/integral_constant.hpp>

namespace gridtools {

    /*
     * @struct is_meta_predicate
     * Check if it yelds true_type or false_type
     */
    template < typename Pred >
    struct is_meta_predicate : boost::false_type {};

    template <>
    struct is_meta_predicate< boost::true_type > : boost::true_type {};

    template <>
    struct is_meta_predicate< boost::false_type > : boost::true_type {};
}
