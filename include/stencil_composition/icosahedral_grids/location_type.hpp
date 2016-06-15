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
#include <common/string_c.hpp>
namespace gridtools {
    template < int I, ushort_t NColors >
    struct location_type {
        static const int value = I;
        using n_colors = static_ushort< NColors >; //! <- is the number of locations of this type
    };

    template < typename T >
    struct is_location_type : boost::mpl::false_ {};

    template < int I, ushort_t NColors >
    struct is_location_type< location_type< I, NColors > > : boost::mpl::true_ {};

    template < int I, ushort_t NColors >
    std::ostream &operator<<(std::ostream &s, location_type< I, NColors >) {
        return s << "location_type<" << I << "> with " << NColors << " colors";
    }
} // namespace gridtools
