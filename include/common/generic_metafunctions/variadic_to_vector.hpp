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
#include <boost/mpl/vector.hpp>
#include <boost/mpl/push_back.hpp>

namespace gridtools {

#ifdef CXX11_ENABLED
    /**
     * @struct variadic_to_vector
     * metafunction that returns a mpl vector from a pack of variadic arguments
     * This is a replacement of using type=boost::mpl::vector<Args ...>, but at the moment nvcc
     * does not properly unpack the last arg of Args... when building the vector. We can eliminate this
     * metafunction once the vector<Args...> works
     */
    template < typename... Args >
    struct variadic_to_vector;

    template < class T, typename... Args >
    struct variadic_to_vector< T, Args... > {
        typedef typename boost::mpl::push_front< typename variadic_to_vector< Args... >::type, T >::type type;
    };

    template < class T >
    struct variadic_to_vector< T > {
        typedef boost::mpl::vector< T > type;
    };

    template <>
    struct variadic_to_vector<> {
        typedef boost::mpl::vector<> type;
    };

#endif
}
