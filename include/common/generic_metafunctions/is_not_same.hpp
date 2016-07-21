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
#include <boost/mpl/not.hpp>

namespace gridtools {

    /*
    * @struct is_not_same
    * just a not of is_same
    */
    template < typename T1, typename T2 >
    struct is_not_same {
        typedef typename boost::mpl::not_< typename boost::is_same< T1, T2 >::type >::type type;
        BOOST_STATIC_CONSTANT(bool, value = (type::value));
    };
}
