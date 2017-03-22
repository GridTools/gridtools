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

#pragma once

#ifndef _USE_GPU_
#error This example requires GPUs
#endif
#ifndef STRUCTURED_GRIDS
#error This example only works on structured grids for now
#endif

#include <iostream>
#include "stencil-composition/interval.hpp"
#include "stencil-composition/level.hpp"
#include "stencil-composition/grid.hpp"

extern int size_cpp;
extern int size_cu;
extern int v1_cu;
extern int v2_cu;

using gridtools::level;
typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

using grid_type = gridtools::grid< axis >;

extern grid_type my_grid;

struct linking_issue {
    grid_type const &
        my_grid; // we would need a copy constructor for gpu_clonable object, this copy would lead to double destruction
    int my_var1;
    int my_var2;

    linking_issue(const grid_type &my_grid);
    void member_foo();
};
