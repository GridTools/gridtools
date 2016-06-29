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
#ifndef _NUMERICS_H_
#define _NUMERICS_H_

/**
@file
@brief compile-time computation of the power three.
*/

namespace gridtools {
    namespace _impl {
        /** @brief Compute 3^I at compile time*/
        template < uint_t I >
        struct static_pow3;

        template <>
        struct static_pow3< 1 > {
            static const int value = 3;
        };

        template < uint_t I >
        struct static_pow3 {
            static const int value = 3 * static_pow3< I - 1 >::value;
        };
    }
}

#endif
