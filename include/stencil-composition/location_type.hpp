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
    template < int I, ushort_t NColors >
    struct location_type {
        static const int value = I;
        using n_colors = static_ushort< NColors >; //! <- is the number of locations of this type
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
    using cells_index = static_int< 0 >;
    using edges_index = static_int< 1 >;
    using vertexes_index = static_int< 2 >;

    using cells = location_type< cells_index::value, 2 >;
    using edges = location_type< edges_index::value, 3 >;
    using vertexes = location_type< vertexes_index::value, 1 >;

    }

    template < uint_t Index >
    struct get_location_by_metastorage_index {
        typedef location_type< -1, 1 > type;
    };

    template <>
    struct get_location_by_metastorage_index< enumtype::cells_index::value + enumtype::metastorage_library_indices_limit > {
        typedef enumtype::cells type;
    };

    template <>
    struct get_location_by_metastorage_index< enumtype::cells_index::value + enumtype::metastorage_library_indices_limit * 2 > {
        typedef enumtype::cells type;
    };

    template <>
    struct get_location_by_metastorage_index< enumtype::edges_index::value + enumtype::metastorage_library_indices_limit > {
        typedef enumtype::edges type;
    };
    template <>
    struct get_location_by_metastorage_index< enumtype::edges_index::value + enumtype::metastorage_library_indices_limit * 2 > {
        typedef enumtype::edges type;
    };

    template <>
    struct get_location_by_metastorage_index< enumtype::vertexes_index::value + enumtype::metastorage_library_indices_limit > {
        typedef enumtype::vertexes type;
    };
    template <>
    struct get_location_by_metastorage_index< enumtype::vertexes_index::value + enumtype::metastorage_library_indices_limit * 2 > {
        typedef enumtype::vertexes type;
    };
} // namespace gridtools
