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
#include <boost/mpl/copy.hpp>
#include <boost/mpl/inserter.hpp>
#include <boost/mpl/insert.hpp>

namespace gridtools {
    // similar to boost::mpl::copy but it copies into an associative set container
    template < typename ToInsert, typename Seq >
    struct copy_into_set {
        typedef typename boost::mpl::copy< ToInsert,
            boost::mpl::inserter< boost::mpl::set0<>, boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > > >::type
            type;
    };

} // namespace
