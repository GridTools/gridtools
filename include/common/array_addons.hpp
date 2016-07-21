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
#include <common/array.hpp>

namespace gridtools {
    template < typename T, size_t D >
    std::ostream &operator<<(std::ostream &s, array< T, D > const &a) {
        s << " {  ";
        for (int i = 0; i < D - 1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D - 1] << "  } ";

        return s;
    }

} // namespace gridtools

template < typename T, typename U, size_t D >
bool same_elements(gridtools::array< T, D > const &a, gridtools::array< U, D > const &b) {
    // shortcut
    if (a.size() != b.size())
        return false;

    // sort and check for equivalence
    gridtools::array< T, D > a0 = a;
    gridtools::array< U, D > b0 = b;
    std::sort(a0.begin(), a0.end());
    std::sort(b0.begin(), b0.end());
    return std::equal(a0.begin(), a0.end(), b0.begin());
}

template < typename T, typename U, size_t D >
bool operator==(gridtools::array< T, D > const &a, gridtools::array< U, D > const &b) {
    return std::equal(a.begin(), a.end(), b.begin());
}
