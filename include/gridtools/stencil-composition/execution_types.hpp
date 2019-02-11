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
#include <type_traits>

#define GT_DEFAULT_VERTICAL_BLOCK_SIZE 20

namespace gridtools {
    namespace execute {
        template <uint_t BlockSize>
        struct parallel_block {
            static const uint_t block_size = BlockSize;
        };
        using parallel = parallel_block<GT_DEFAULT_VERTICAL_BLOCK_SIZE>;
        struct forward {};
        struct backward {};

        template <typename T>
        struct is_parallel : std::false_type {};

        template <uint_t BlockSize>
        struct is_parallel<parallel_block<BlockSize>> : std::true_type {};

        template <typename T>
        struct is_forward : std::false_type {};

        template <uint_t BlockSize>
        struct is_forward<forward> : std::true_type {};

        template <typename T>
        struct is_backward : std::false_type {};

        template <uint_t BlockSize>
        struct is_backward<backward> : std::true_type {};
    } // namespace execute

    template <typename T>
    struct is_execution_engine : std::false_type {};

    template <uint_t BlockSize>
    struct is_execution_engine<execute::parallel_block<BlockSize>> : std::true_type {};

    template <>
    struct is_execution_engine<execute::forward> : std::true_type {};

    template <>
    struct is_execution_engine<execute::backward> : std::true_type {};

} // namespace gridtools
