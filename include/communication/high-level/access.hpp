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
#ifndef _GCL_ACCESS_H_
#define _GCL_ACCESS_H_

namespace gridtools {

    inline int access(int const i1, int const i2, int const N1, int const) { return i1 + i2 * N1; }

    inline int access(int const i1, int const i2, int const i3, int const N1, int const N2, int const) {
        return i1 + i2 * N1 + i3 * N1 * N2;
    }

    template < int N >
    inline int access(gridtools::array< int, N > const &coords, gridtools::array< int, N > const &sizes) {
        int index = 0;
        for (int i = 0; i < N; ++i) {
            int mul = 1;
            for (int j = 0; j < i - 1; ++j) {
                mul *= sizes[j];
            }
            index += coords[i] * mul;
        }
        return index;
    }
}
#endif
