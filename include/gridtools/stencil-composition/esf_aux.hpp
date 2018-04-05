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
#include "./arg.hpp"
#include "accessor_fwd.hpp"

namespace gridtools {

    namespace impl {
        /** metafunction that associates (in a mpl::map) placeholders to extents.
        * It returns a mpl::map between placeholders and extents of the local arguments.
        */
        template < typename Placeholders, typename Accessors >
        struct make_arg_with_extent_map {

            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Placeholders, is_arg >::value),
                "The list of Placeholders seems to contain elements that are not placeholers (i.e., they are not of "
                "type arg)");
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Accessors, is_accessor >::value), GT_INTERNAL_ERROR);
#ifdef PEDANTIC // with global accessors this assertion fails (since they are not in the Accessors)
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Placeholders >::value == boost::mpl::size< Accessors >::value),
                "Size of placeholder arguments passed to esf \n"
                "    make_stage<functor>(arg1(), arg2()) )\n"
                "does not match the list of arguments defined within the ESF, like\n"
                "    typedef boost::mpl::vector2<arg_in, arg_out> arg_list.");
#endif
            template < typename Accessor >
            struct _get_extent {
                typedef typename Accessor::extent_t type;
            };

            /** Given the list of placeholders (Plcs) and the list of arguemnts of a
                stencil operator (Accessors), this struct will insert the placeholder type
                (as key) and the corresponding extent into an mpl::map.
            */
            template < typename Plcs, typename LArgs >
            struct from {
                template < typename CurrentMap, typename Index >
                struct insert {

                    typedef typename boost::mpl::at_c< LArgs, Index::value >::type accessor_t;
                    typedef typename boost::mpl::insert< CurrentMap,
                        typename boost::mpl::pair< typename boost::mpl::at_c< Plcs, Index::value >::type,
                                                             typename _get_extent< accessor_t >::type > >::type type;
                };
            };

            // Note: only the accessors of storage type are considered in the sequence
            typedef typename boost::mpl::range_c< uint_t, 0, boost::mpl::size< Accessors >::type::value > iter_range;

            /** Here the iteration begins by filling an empty map */
            typedef typename boost::mpl::fold< iter_range,
                boost::mpl::map0<>,
                typename from< Placeholders, Accessors >::template insert< boost::mpl::_1, boost::mpl::_2 > >::type
                type;
        };
    }
}
