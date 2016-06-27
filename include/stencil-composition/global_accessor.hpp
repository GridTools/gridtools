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

#include "../common/defs.hpp"
#include "./empty_extent.hpp"

namespace gridtools {

    template < uint_t I, enumtype::intend Intend >
    struct global_accessor {

        typedef global_accessor< I, Intend > type;

        typedef static_uint< I > index_type;

        typedef empty_extent extent_t;
    };

    template < typename Type >
    struct is_global_accessor : boost::false_type {};

    template < uint_t I, enumtype::intend Intend >
    struct is_global_accessor< global_accessor< I, Intend > > : boost::true_type {};

} // namespace gridtools
