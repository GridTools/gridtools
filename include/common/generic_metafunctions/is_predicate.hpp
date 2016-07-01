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
