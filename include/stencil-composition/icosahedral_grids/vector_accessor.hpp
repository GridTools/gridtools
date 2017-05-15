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
#include "../accessor_fwd.hpp"
#include "../extent.hpp"
/** @file vector accessor */

namespace gridtools {

    /**
       @brief accessor for an expandable parameters list

       accessor object used with the expandable parameters. It is exactly like a regular accessor.
       Its type must be different though, so that the gridtools::iterate_domain can implement a specific
       overload of the operator() for this accessor type.

       \tparam ID integer identifier, to univocally specify the accessor
       \tparam Intent flag stating wether or not this accessor is read only
       \tparam Extent specification of the minimum box containing the stencil access pattern
       \tparam NDim dimensionality of the vector accessor: should be the storage space dimensions plus two (the vector
       field/snapshot dimensions). In case of icosahedral grids for a 3D field this dimension is 6 (3 space dimensions +
       1 color + 2 field/snapshot dimensions)
    */
    template < uint_t ID,
        enumtype::intend Intent,
        typename LocationType,
        typename Extent = extent< 0 >,
        ushort_t NDim = 6 >
    struct vector_accessor : accessor< ID, Intent, LocationType, Extent, NDim > {

#ifdef CXX11_ENABLED
        using super = accessor< ID, Intent, LocationType, Extent, NDim >;
        using super::accessor;
        static const ushort_t n_dimensions = NDim;
#else
        GRIDTOOLS_STATIC_ASSERT(
            NDim > 0, "You are using a vector_accessor and compiling with C++03, switch to C++11 (-DENABLE_CXX11=ON)");
#endif
    };

    template < typename T >
    struct is_vector_accessor : boost::mpl::false_ {};

    template < uint_t ID, enumtype::intend Intent, typename LocationType, typename Extent, uint_t Size >
    struct is_vector_accessor< vector_accessor< ID, Intent, LocationType, Extent, Size > > : boost::mpl::true_ {};

    template < typename T >
    struct is_any_accessor : boost::mpl::or_< is_accessor< T >, is_vector_accessor< T > > {};

} // namespace gridtools
