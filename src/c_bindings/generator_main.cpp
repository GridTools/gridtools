/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <fstream>
#include <iostream>
#include <string>

#include "c_bindings/generator.hpp"

namespace {
    std::string stem(const std::string &src) {
        auto last_before_trailing_slashes = src.find_last_not_of('/');
        if (last_before_trailing_slashes == std::string::npos)
            return "";
        auto end = last_before_trailing_slashes + 1;
        if (end == src.size())
            end = std::string::npos;
        auto last_slash = src.rfind('/', end);
        auto begin = last_slash == std::string::npos ? 0 : last_slash + 1;
        return src.substr(begin, src.find_first_of("./", begin));
    }
}

int main(int argc, const char *argv[]) {
    if (argc > 2) {
        std::ofstream dst(argv[2]);
        auto module = argc > 3 ? argv[3] : stem(argv[2]);
        gridtools::c_bindings::generate_fortran_interface(dst, module);
    }
    if (argc > 1) {
        std::ofstream dst(argv[1]);
        gridtools::c_bindings::generate_c_interface(dst);
    } else {
        gridtools::c_bindings::generate_c_interface(std::cout);
    }
}
