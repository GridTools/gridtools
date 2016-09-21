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
#include "../extent.hpp"
#include "../accessor_base.hpp"

namespace gridtools {
    /**
    * This is the type of the accessors accessed by a stencil functor.
    * It's a pretty minima implementation.
    */
    template < uint_t ID,
        enumtype::intend Intend,
        typename LocationType,
        typename Extent = extent< 0,0,0,0,0,0>,
        ushort_t FieldDimensions = 4 >
    struct accessor : public accessor_base< ID, Intend, Extent, FieldDimensions > {
        GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "Error: wrong type");
        using type = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;
        using location_type = LocationType;
        static const uint_t value = ID;
        using index_type = static_uint< ID >;
        using extent_t = Extent;
        location_type location() const { return location_type(); }

        typedef accessor_base< ID, Intend, Extent, FieldDimensions > super;

        GT_FUNCTION
        constexpr accessor() : super() {}

    /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;

        GT_FUNCTION
        constexpr explicit accessor(array< int_t, FieldDimensions > const &offsets) : super(offsets) {}
    };

    template < uint_t ID, typename LocationType, typename Extent = extent< 0 >, ushort_t FieldDimensions = 4 >
    using in_accessor = accessor< ID, enumtype::in, LocationType, Extent, FieldDimensions >;

    template < uint_t ID, typename LocationType, ushort_t FieldDimensions = 4 >
    using inout_accessor = accessor< ID, enumtype::inout, LocationType, extent< 0 >, FieldDimensions >;

    template < typename T >
    struct is_accessor : boost::mpl::false_ {};

    template < uint_t ID, enumtype::intend Intend, typename LocationType, typename Extent, ushort_t FieldDimensions >
    struct is_accessor< accessor< ID, Intend, LocationType, Extent, FieldDimensions > > : boost::mpl::true_ {};

} // namespace gridtools
