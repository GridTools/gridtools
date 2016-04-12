#pragma once
#include "../common/defs.hpp"
namespace gridtools {

    template < typename T >
    struct is_computation_token : boost::mpl::or_< boost::mpl::or_< is_condition< T >, is_mss_descriptor< T > >,
                                      is_reduction_descriptor< T > >::type {};
}
