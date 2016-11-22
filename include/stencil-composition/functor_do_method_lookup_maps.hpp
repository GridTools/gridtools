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

#include <boost/static_assert.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/if.hpp>
#include "level.hpp"
#include "interval.hpp"

namespace gridtools {
    // implementation of the do method lookup map computation

    /**
     * @struct is_sub_interval
     * Meta function returning true if the index pair 1 is contained in the index pair 2
     */
    template < typename TIndexPair1, typename TIndexPair2 >
    struct is_sub_interval : boost::mpl::bool_< TIndexPair1::first::value >= TIndexPair2::first::value &&
                                                TIndexPair1::second::value <= TIndexPair2::second::value > {};

    /**
     * @struct do_method_lookup_map_add
     * Meta function updating a do method lookup map iterator pair given the next loop interval
     * (the iterator points to the currently active do method and is updated while we iterate over the loop intervals)
     */
    template < typename TDoMethodLookupMap, typename TLoopInterval >
    struct do_method_lookup_map_add {
        // extract the do method lookup map and the do method iterator
        typedef typename boost::mpl::first< TDoMethodLookupMap >::type DoMethodLookupMap;
        typedef typename boost::mpl::second< TDoMethodLookupMap >::type DoIterator;

        // move the do method iterator forward until the associated do method includes the current loop interval

        typedef
            typename boost::mpl::if_< is_sub_interval< TLoopInterval, typename boost::mpl::deref< DoIterator >::type >,
                DoIterator,
                typename boost::mpl::next< DoIterator >::type >::type NextDoIterator;
        typedef typename boost::mpl::deref< NextDoIterator >::type NextDoInterval;

        // check that the computed do method includes the loop interval
        GRIDTOOLS_STATIC_ASSERT((is_sub_interval< TLoopInterval, NextDoInterval >::value),
            "check that the computed do method includes the loop interval");

        // add the loop interval to the do method lookup map
        // (only add empty loop intervals if the functor loop interval is empty)
        typedef
            typename boost::mpl::insert< DoMethodLookupMap,
                boost::mpl::pair< TLoopInterval,
                                             typename make_interval< typename boost::mpl::first< NextDoInterval >::type,
                                                 typename boost::mpl::second< NextDoInterval >::type >::type > >::type
                NextDoMethodLookupMap;

        // return a do method lookup map iterator pair
        typedef boost::mpl::pair< NextDoMethodLookupMap, NextDoIterator > type;
    };

    /**
     * @struct compute_functor_do_method_lookup_map
     * Meta function computing a do method lookup map for a specific functor
     * (the map associates a do method interval to every loop interval)
     */
    template < typename TDoMethods, typename TLoopIntervals >
    struct compute_functor_do_method_lookup_map {
        // check the collection is not empty
        GRIDTOOLS_STATIC_ASSERT(
            !boost::mpl::empty< TDoMethods >::value, " check that the collection of Do methods is not empty");
        GRIDTOOLS_STATIC_ASSERT(
            !boost::mpl::empty< TLoopIntervals >::value, " check that the collection of loop intervals is not empty");

        // compute the extentd spanned by all do methods
        typedef boost::mpl::pair< typename boost::mpl::front< TDoMethods >::type::first,
            typename boost::mpl::back< TDoMethods >::type::second > DoExtend;

        // compute the extent spanned by all loop intervals
        typedef boost::mpl::pair< typename boost::mpl::front< TLoopIntervals >::type::first,
            typename boost::mpl::back< TLoopIntervals >::type::second > LoopExtend;

        // make sure the do extent is a sub interval of the loop extent
        GRIDTOOLS_STATIC_ASSERT((is_sub_interval< DoExtend, LoopExtend >::value),
            "make sure the do extent is a sub interval of the loop extent");

        // extract all loop intervals inside the functor do method interval
        typedef typename boost::mpl::copy_if< TLoopIntervals, is_sub_interval< boost::mpl::_, DoExtend > >::type
            LoopIntervals;

        // check there is a loop interval for every functor
        GRIDTOOLS_STATIC_ASSERT(
            boost::mpl::size< LoopIntervals >::value > 0, "check there is a loop interval for every functor");

        // iterate over all loop intervals and compute the do method lookup map
        // (the state of the fold operation contains the map as well as an iterator
        // pointing to the currently active do method which includes the current loop interval)
        typedef typename boost::mpl::fold< LoopIntervals,
            boost::mpl::pair< boost::mpl::map0<>, typename boost::mpl::begin< TDoMethods >::type >,
            do_method_lookup_map_add< boost::mpl::_1, boost::mpl::_2 > >::type DoMethodLookupMap;

        // remove the do method iterator and return the map only
        typedef typename boost::mpl::first< DoMethodLookupMap >::type type;
    };
} // namespace gridtools
