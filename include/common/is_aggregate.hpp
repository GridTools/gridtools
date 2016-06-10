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

#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_integral.hpp>
#include "array.hpp"

namespace gridtools {

    /**
     * type trait to check if a type is an aggregate
     * Note: see discussion here
     * http://stackoverflow.com/questions/33648044/boostprotois-aggregate-returning-false-when-it-is-an-aggregate-type
     * there is not general way of detecting whether a type in C++ is an aggregate, and there probably wont be.
     * Instead we use specific traits for the types that are used in our library
     * (in the future this might be extended to using concepts)
     */
    template < typename T >
    struct is_aggregate : boost::mpl::or_< is_array< T >, boost::is_integral< T > > {};
}
