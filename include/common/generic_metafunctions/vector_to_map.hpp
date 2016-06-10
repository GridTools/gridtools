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
#include <boost/mpl/map.hpp>
#include <boost/fusion/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/fusion/algorithm/transformation/insert.hpp>
#include <boost/fusion/include/insert.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>

namespace gridtools {

    /**
     * @struct vector_to_map
     * convert a vector of pairs into a make_pair
     */
    template < typename Vec >
    struct vector_to_map {
        typedef typename boost::mpl::fold< Vec,
            boost::mpl::map0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
