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
#include "accumulate.hpp"
namespace gridtools {
#ifdef CXX11_ENABLED

    /**
       SFINAE for the case in which all the components of a parameter pack are of integral type
    */
    template < typename... IntTypes >
    using all_integers =
        typename boost::enable_if_c< accumulate(logical_and(), boost::is_integral< IntTypes >::type::value...),
            bool >::type;

    /**
       SFINAE for the case in which all the components of a parameter pack are of integral type
    */
    template <typename ... IntTypes>
    using not_all_integers=typename boost::disable_if_c<accumulate(logical_and(),  boost::is_integral<IntTypes>::type::value ... ), bool >::type;

    /**
       SFINAE for the case in which all the components of a parameter pack are of static integral type
    */
    template < typename... IntTypes >
    using all_static_integers =
        typename boost::enable_if_c< accumulate(logical_and(), is_static_integral< IntTypes >::type::value...),
            bool >::type;

#endif
}
