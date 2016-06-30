#pragma once
#include "case_type.hpp"
/**@file*/

namespace gridtools {
    /**@brief interface for specifying a case from whithin a @ref gridtools::switch_ statement
     */
    template < typename T, typename Mss >
    case_type< T, Mss > case_(T val_, Mss mss_) {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token< Mss >::value), "wrong type");
        return case_type< T, Mss >(val_, mss_);
    }

    /**@brief interface for specifying a default case from whithin a @ref gridtools::switch_ statement
     */
    template < typename Mss >
    default_type< Mss > default_(Mss mss_) {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token< Mss >::value), "wrong type");
        return default_type< Mss >(mss_);
    }
} // namespace gridtools
