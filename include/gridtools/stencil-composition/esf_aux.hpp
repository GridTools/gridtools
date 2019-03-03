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

#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/size.hpp>

#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "arg.hpp"
#include "is_accessor.hpp"

namespace gridtools {

    namespace impl {
        /** metafunction that associates (in a mpl::map) placeholders to extents.
         * It returns a mpl::map between placeholders and extents of the local arguments.
         */
        template <typename Placeholders, typename Accessors>
        struct make_arg_with_extent_map {

            GT_STATIC_ASSERT((is_sequence_of<Placeholders, is_plh>::value),
                "The list of Placeholders seems to contain elements that are not placeholers (i.e., they are not of "
                "type arg)");
            GT_STATIC_ASSERT((is_sequence_of<Accessors, is_accessor>::value), GT_INTERNAL_ERROR);
#ifdef GT_PEDANTIC // with global accessors this assertion fails (since they are not in the Accessors)
            GT_STATIC_ASSERT(boost::mpl::size<Placeholders>::value == boost::mpl::size<Accessors>::value,
                "Size of placeholder arguments passed to esf \n"
                "    make_stage<functor>(arg1(), arg2()) )\n"
                "does not match the list of arguments defined within the ESF, like\n"
                "    typedef boost::mpl::vector2<arg_in, arg_out> param_list.");
#endif

            /** Given the list of placeholders (Plcs) and the list of arguemnts of a
                stencil operator (Accessors), this struct will insert the placeholder type
                (as key) and the corresponding extent into an mpl::map.
            */
            template <typename CurrentMap, typename Index>
            struct do_insert
                : boost::mpl::insert<CurrentMap,
                      typename boost::mpl::pair<typename boost::mpl::at_c<Placeholders, Index::value>::type,
                          typename boost::mpl::at_c<Accessors, Index::value>::type::extent_t>> {};

            // Note: only the accessors of storage type are considered in the sequence
            typedef typename boost::mpl::range_c<uint_t, 0, boost::mpl::size<Accessors>::type::value> iter_range;

            /** Here the iteration begins by filling an empty map */
            typedef typename boost::mpl::
                fold<iter_range, boost::mpl::map0<>, do_insert<boost::mpl::_1, boost::mpl::_2>>::type type;
        };
    } // namespace impl
} // namespace gridtools
