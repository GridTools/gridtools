/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
