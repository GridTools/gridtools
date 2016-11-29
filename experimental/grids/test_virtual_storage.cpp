/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include <iostream>
#include <common/defs.h>
#include "virtual_storage.hpp"
#include "array_addons.hpp"

using gridtools::uint_t;

int main() {
    gridtools::virtual_storage< gridtools::layout_map< 0, 1, 2 > > v_storage(gridtools::array< uint_t, 3 >(34, 12, 17));

    for (int i = 0; i < v_storage.dims< 0 >(); ++i) {
        for (int j = 0; j < v_storage.dims< 1 >(); ++j) {
            for (int k = 0; k < v_storage.dims< 2 >(); ++k) {
                bool result =
                    v_storage.offset2indices(v_storage._index(i, j, k)) == gridtools::array< int, 3 >{i, j, k};
                if (!result) {
                    std::cout << "Error: index = " << v_storage._index(i, j, k) << ", "
                              << v_storage.offset2indices(v_storage._index(i, j, k))
                              << " != " << gridtools::array< int, 3 >{i, j, k} << std::endl;
                    std::cout << std::boolalpha << result << std::endl;
                }
                assert(result);
            }
        }
    }
    std::cout << "SUCCESS!" << std::endl;
    return 0;
}
