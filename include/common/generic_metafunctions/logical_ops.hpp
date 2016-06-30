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
#include "../host_device.hpp"

namespace gridtools {

    /**@brief operation to be used inside the accumulator*/
    struct logical_and {
        GT_FUNCTION
        constexpr logical_and() {}
        template < typename T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x && y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct logical_or {
        GT_FUNCTION
        constexpr logical_or() {}
        template < typename T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x || y;
        }
    };
}
