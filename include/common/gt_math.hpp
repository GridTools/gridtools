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

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <uint_t Number>
    struct gt_pow{
        template<typename Value>
        GT_FUNCTION
        static Value constexpr apply(Value const& v)
            {
                return v*gt_pow<Number-1>::apply(v);
            }
    };

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <>
    struct gt_pow< 0 > {
        template < typename Value >
        GT_FUNCTION static Value constexpr apply(Value const &v) {
            return 1.;
        }
    };
} // namespace gridtools
