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

#include <boost/mpl/equal.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set/set0.hpp>

#include "../../common/generic_metafunctions/is_there_in_sequence_if.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "./esf.hpp"

namespace gridtools {
    namespace icgrid {
        template <typename EsfSequence>
        struct extract_esf_location_type {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value),
                GT_INTERNAL_ERROR_MSG("Error, wrong esf types"));
            typedef typename boost::mpl::fold<EsfSequence,
                boost::mpl::set0<>,
                boost::mpl::insert<boost::mpl::_1, esf_get_location_type<boost::mpl::_2>>>::type location_type_set_t;
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<location_type_set_t>::value == 1),
                "Error: all ESFs in a Multi Stage stencil should have the same location type");

            typedef typename boost::mpl::front<location_type_set_t>::type type;
        };

        /**
         * metafunction that returns true if: "at least one esf of the sequence has a color that matches
         * the Color parameter or the color of the esf is specified as nocolor
         * (meaning that any color should be matched)
         * @tparam EsfSequence sequence of esfs
         * @tparam Color color to be matched by the ESFs
         */
        template <typename EsfSequence, typename Color>
        struct esf_sequence_contains_color {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_color_type<Color>::value), GT_INTERNAL_ERROR);

            template <typename Esf>
            struct esf_has_color_ {
                GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), GT_INTERNAL_ERROR);
                typedef static_bool<(boost::is_same<typename Esf::color_t::color_t, typename Color::color_t>::value ||
                                     boost::is_same<typename Esf::color_t, nocolor>::value)>
                    type;
            };

            typedef typename is_there_in_sequence_if<EsfSequence, esf_has_color_<boost::mpl::_>>::type type;
        };
    } // namespace icgrid

    template <typename Esf1, typename Esf2>
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf1>::value && is_esf_descriptor<Esf2>::value), GT_INTERNAL_ERROR);
        typedef static_bool<boost::is_same<typename Esf1::esf_function, typename Esf2::esf_function>::value &&
                            boost::mpl::equal<typename Esf1::args_t, typename Esf2::args_t>::value &&
                            boost::is_same<typename Esf1::location_type, typename Esf2::location_type>::value &&
                            boost::is_same<typename Esf1::grid_t, typename Esf2::grid_t>::value>
            type;
    };

    template <typename Esf>
    struct esf_arg_list {
        template <typename Set, typename Item>
        struct insert_arglist {
            typedef
                typename boost::mpl::insert<Set, typename Esf::template esf_function<Item::value>::arg_list>::type type;
        };

        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), GT_INTERNAL_ERROR);

        typedef typename boost::mpl::fold<boost::mpl::range_c<uint_t, 0, Esf::location_type::n_colors::value>,
            boost::mpl::set0<>,
            insert_arglist<boost::mpl::_1, boost::mpl::_2>>::type checkset;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<checkset>::value == 1),
            "Multiple Color specializations of the same ESF must contain the same arg list");

        typedef typename Esf::template esf_function<0>::arg_list type;
    };

    /** Retrieve the extent in esf_descriptor_with_extents

       \tparam Esf The esf_descriptor that must be the one speficying the extent
    */
    template <typename Esf>
    struct esf_extent;

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct esf_extent<esf_descriptor_with_extent<Functor, Grid, LocationType, Extent, Color, ArgSequence>> {
        using type = Extent;
    };

    GT_META_LAZY_NAMESPASE {
        template <class Esf, class Args>
        struct esf_replace_args;

        template <template <uint_t> class F, class Grid, class Location, class Color, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor<F, Grid, Location, Color, OldArgs>, NewArgs> {
            using type = esf_descriptor<F, Grid, Location, Color, NewArgs>;
        };

        template <template <uint_t> class F,
            class Grid,
            class Location,
            class Extent,
            class Color,
            class OldArgs,
            class NewArgs>
        struct esf_replace_args<esf_descriptor_with_extent<F, Grid, Location, Extent, Color, OldArgs>, NewArgs> {
            using type = esf_descriptor_with_extent<F, Grid, Location, Extent, Color, NewArgs>;
        };
    }
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
