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

#include "gtest/gtest.h"
#include <iostream>
#include "stencil-composition/interval.hpp"
#include "stencil-composition/level.hpp"
#include "stencil-composition/grid.hpp"
#include "linking_issue.hpp"

int size_cpp;

TEST(linking_issue, test) {
    my_grid.value_list[0] = 1;
    my_grid.value_list[1] = 2;
    // std::cout << "in .cpp:" << std::endl;
    // std::cout << "sizeof grid: " << sizeof(my_grid) << std::endl;
    size_cpp = sizeof(my_grid);

    linking_issue tmp(my_grid);

    //    std::cout << "=== after assign" << std::endl;
    tmp.my_var1 = 11;
    tmp.my_var2 = 12;
    // std::cout << "in .cpp:" << std::endl;

    // std::cout << tmp.my_var1 << std::endl;
    // std::cout << tmp.my_var2 << std::endl;

    tmp.member_foo();

    EXPECT_TRUE(v1_cu == tmp.my_var1);
    EXPECT_TRUE(v2_cu == tmp.my_var2);

    // std::cout << size_cu << ", " << size_cpp << "\n";

    EXPECT_TRUE(size_cu == size_cpp);
}
