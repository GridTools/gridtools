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
#include "esf.hpp"

namespace gridtools {

    template < uint_t T, uint_t SwitchId >
    struct conditional;

    template < typename EsfSequence >
    struct independent_esf {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value),
            "Error: independent_esf requires a sequence of esf's");
        typedef EsfSequence esf_list;
    };

    template < typename T >
    struct is_independent : boost::false_type {};

    template < typename T >
    struct is_independent< independent_esf< T > > : boost::true_type {};

} // namespace gridtools
