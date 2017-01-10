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

#include "defs.hpp"
#include "nano_array.hpp"
#include "storage_info_metafunctions.hpp"

namespace gridtools {

    template < unsigned N >
    struct alignment {
        const static unsigned value = N;
    };

    /* aligned storage infos use this struct */
    template < typename Alignment, typename Layout, typename Halo >
    struct alignment_impl;

    template < unsigned M, typename Layout, typename Halo >
    struct alignment_impl< alignment< M >, Layout, Halo > {
        static constexpr unsigned N = Layout::length;
        static constexpr unsigned InitialOffset = get_initial_offset< Layout, alignment< M >, Halo >::compute();

        nano_array< unsigned, N > m_unaligned_dims;
        nano_array< unsigned, N > m_unaligned_strides;

        constexpr alignment_impl(nano_array< unsigned, N > dims, nano_array< unsigned, N > strides)
            : m_unaligned_dims(dims), m_unaligned_strides(strides) {}

        template < unsigned Coord >
        GT_FUNCTION unsigned unaligned_dim() {
            return m_unaligned_dims.at[Coord];
        }

        template < unsigned Coord >
        GT_FUNCTION unsigned unaligned_stride() {
            return m_unaligned_strides[Coord];
        }
    };

    /* specialization for unaligned storage infos */
    template < typename Layout, typename Halo >
    struct alignment_impl< alignment< 0 >, Layout, Halo > {
        static constexpr unsigned InitialOffset = 0;
        template < typename... T >
        constexpr alignment_impl(T... t) {}
    };
}
