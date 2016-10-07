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
#include <common/string_c.hpp>
#include "../common/defs.hpp"

namespace gridtools {
    template < int_t I, ushort_t NColors >
    struct location_type {
        static const int_t value = I;
        typedef static_ushort< NColors > n_colors; //! <- is the number of locations of this type
    };

    template < typename T >
    struct is_location_type : boost::mpl::false_ {};

    template < int I, ushort_t NColors >
    struct is_location_type< location_type< I, NColors > > : boost::mpl::true_ {};

    template < int I, ushort_t NColors >
    std::ostream &operator<<(std::ostream &s, location_type< I, NColors >) {
        return s << "location_type<" << I << "> with " << NColors << " colors";
    }

    namespace enumtype {
        typedef static_int< 0 > cells_index;
        typedef static_int< 1 > edges_index;
        typedef static_int< 2 > vertexes_index;

        typedef location_type< cells_index::value, 2 > cells;
        typedef location_type< edges_index::value, 3 > edges;
        typedef location_type< vertexes_index::value, 1 > vertexes;
    }

} // namespace gridtools
