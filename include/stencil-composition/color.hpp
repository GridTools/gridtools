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
    template < uint_t c >
    struct color_type {
        typedef static_uint< c > color_t;
    };

    struct nocolor {
        typedef notype color_t;
    };

    template < typename T >
    struct is_color_type : boost::mpl::false_ {};

    template < uint_t c >
    struct is_color_type< color_type< c > > : boost::mpl::true_ {};

    template <>
    struct is_color_type< nocolor > : boost::mpl::true_ {};
};
