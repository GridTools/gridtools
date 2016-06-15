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

namespace gridtools {
    namespace _impl {
        /**@brief wrap type to simplify specialization based on mpl::vectors */
        template < typename MplArray >
        struct wrap_type {
            typedef MplArray type;
        };

        /**
     * @brief compile-time boolean operator returning true if the template argument is a wrap_type
     * */
        template < typename T >
        struct is_wrap_type : boost::false_type {};

        template < typename T >
        struct is_wrap_type< wrap_type< T > > : boost::true_type {};
    }
}
