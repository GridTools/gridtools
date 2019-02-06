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
#pragma once

#include "../common/defs.hpp"
#include "../stencil-composition/backend.hpp"

#ifdef GT_STRUCTURED_GRIDS
using grid_type_t = gridtools::grid_type::structured;
#else
using grid_type_t = gridtools::grid_type::icosahedral;
#endif

#if FLOAT_PRECISION == 4
using float_type = float;
#elif FLOAT_PRECISION == 8
using float_type = double;
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

#ifdef BACKEND_X86
using target_t = gridtools::target::x86;
#ifdef BACKEND_STRATEGY_NAIVE
using strategy_t = gridtools::strategy::naive;
#else
using strategy_t = gridtools::strategy::block;
#endif
#elif defined(BACKEND_MC)
using target_t = gridtools::target::mc;
using strategy_t = gridtools::strategy::block;
#elif defined(BACKEND_CUDA)
using target_t = gridtools::target::cuda;
using strategy_t = gridtools::strategy::block;
#else
#define NO_BACKEND
#endif

#ifndef NO_BACKEND
using backend_t = gridtools::backend<target_t, grid_type_t, strategy_t>;
#endif
