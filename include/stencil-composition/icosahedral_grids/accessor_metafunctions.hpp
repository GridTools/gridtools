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

#include "../icosahedral_grids/accessor.hpp"
#include "../expandable_parameters/expandable_fwd.hpp"

namespace gridtools {

    template < typename Accessor >
    struct accessor_index {
        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), GT_INTERNAL_ERROR);
        typedef typename Accessor::index_type type;
    };

    template < typename Accessor >
    struct is_accessor_readonly : boost::mpl::false_ {
        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::or_< is_accessor< Accessor >, is_vector_accessor< Accessor > >::value), GT_INTERNAL_ERROR);
    };

    template < uint_t ID, typename LocationType, typename Extent, ushort_t FieldDimensions >
    struct is_accessor_readonly< accessor< ID, enumtype::in, LocationType, Extent, FieldDimensions > >
        : boost::mpl::true_ {};

    /**
     * @brief metafunction that given an accesor and a map, it will remap the index of the accessor according
     * to the corresponding entry in ArgsMap
     */
    template < typename Accessor, typename ArgsMap >
    struct remap_accessor_type {};

    template < uint_t ID,
        enumtype::intend Intend,
        typename LocationType,
        typename Extent,
        ushort_t FieldDimensions,
        typename ArgsMap >
    struct remap_accessor_type< accessor< ID, Intend, LocationType, Extent, FieldDimensions >, ArgsMap > {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), GT_INTERNAL_ERROR_MSG("wrong size"));
        // check that the key type is an int (otherwise the later has_key would never find the key)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same<
                typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                int >::value),
            GT_INTERNAL_ERROR);

        typedef typename boost::mpl::integral_c< int, (int)ID > index_type_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_type_t >::value), GT_INTERNAL_ERROR);

        typedef accessor< boost::mpl::at< ArgsMap, index_type_t >::type::value,
            Intend,
            LocationType,
            Extent,
            FieldDimensions > type;
    };

    template < ushort_t ID, enumtype::intend Intend, typename ArgsMap >
    struct remap_accessor_type< global_accessor< ID, Intend >, ArgsMap > {
        typedef global_accessor< ID, Intend > accessor_t;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), GT_INTERNAL_ERROR_MSG("wrong size"));
        // check that the key type is an int (otherwise the later has_key would never find the key)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same<
                typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                int >::value),
            GT_INTERNAL_ERROR);

        typedef typename boost::mpl::integral_c< int, (int)ID > index_type_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_type_t >::value), GT_INTERNAL_ERROR);

        typedef global_accessor< boost::mpl::at< ArgsMap, index_type_t >::type::value, Intend > type;
    };

} // namespace gridtools
