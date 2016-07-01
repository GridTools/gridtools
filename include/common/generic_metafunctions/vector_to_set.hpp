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
#include <boost/mpl/set/set0.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>

namespace gridtools {

    /**
     * @struct vector_to_map
     * convert a vector sequence into a set sequence
     */
    template < typename Vec >
    struct vector_to_set {
        typedef typename boost::mpl::fold< Vec,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
