/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/mpl/equal.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set/set0.hpp>

#include "../../common/generic_metafunctions/is_there_in_sequence_if.hpp"
#include "../../meta/macros.hpp"
#include "./esf.hpp"

namespace gridtools {
    namespace icgrid {
        template <typename EsfSequence>
        struct extract_esf_location_type {
            GT_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value),
                GT_INTERNAL_ERROR_MSG("Error, wrong esf types"));
            typedef typename boost::mpl::fold<EsfSequence,
                boost::mpl::set0<>,
                boost::mpl::insert<boost::mpl::_1, esf_get_location_type<boost::mpl::_2>>>::type location_type_set_t;
            GT_STATIC_ASSERT(boost::mpl::size<location_type_set_t>::value == 1,
                "Error: all ESFs in a Multi Stage stencil should have the same location type");

            typedef typename boost::mpl::front<location_type_set_t>::type type;
        };
    } // namespace icgrid

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

    GT_META_LAZY_NAMESPACE {
        template <class Esf>
        struct esf_param_list {
            GT_STATIC_ASSERT(is_esf_descriptor<Esf>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(Esf::location_type::n_colors::value > 0, GT_INTERNAL_ERROR);

            template <class I>
            GT_META_DEFINE_ALIAS(nth_param, meta::id, typename Esf::template esf_function<I::value>::param_list);

            using colors_t = GT_META_CALL(meta::make_indices_c, Esf::location_type::n_colors::value);
            using param_lists_t = GT_META_CALL(meta::transform, (nth_param, colors_t));

            GT_STATIC_ASSERT(meta::all_are_same<param_lists_t>::value,
                "Multiple Color specializations of the same ESF must contain the same param list");

            using type = GT_META_CALL(meta::first, param_lists_t);
        };

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
    GT_META_DELEGATE_TO_LAZY(esf_param_list, class Esf, Esf);
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
