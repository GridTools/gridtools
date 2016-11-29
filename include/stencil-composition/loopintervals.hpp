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

#include "interval.hpp"
#include "level.hpp"
#include <boost/mpl/assert.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/prior.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/static_assert.hpp>

namespace gridtools {
    // implementation of the loop interval computation

    /**
     * @struct from_indexes_add_functor_do_methods
     * Meta function adding all from level indexes of a do method vector to the from index set
     * (note that we introduce an additional from index after the last do method interval)
     */
    template < typename TFromIndexSet, typename TDoMethods >
    struct from_indexes_add_functor_do_methods {
        // check there is at least one do method
        BOOST_MPL_ASSERT_MSG((!boost::mpl::empty< TDoMethods >::value),
            NO_FUNCTOR_DO_METHODS_DEFINED_IN_THE_GIVEN_AXIS_INTERVAL,
            (TDoMethods));

        // add all do method vector from level indexes
        typedef typename boost::mpl::fold< TDoMethods,
            TFromIndexSet,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::first< boost::mpl::_2 > > >::type NextFromIndexSet;

        // additionally add a sentinel from index after the last do method interval
        // (note that it is ok to use next as we checked the do method intervals to not max out the offset limits)
        typedef typename boost::mpl::insert< NextFromIndexSet,
            typename boost::mpl::next< typename boost::mpl::second<
                typename boost::mpl::back< TDoMethods >::type >::type >::type >::type type;
    };

    /**
     * @struct compute_loop_intervals
     * Meta function computing a vector of loop intervals given a vector of do method vectors and an axis interval
     */
    template < typename TDoMethods, typename TAxisInterval >
    struct compute_loop_intervals {
        // check the template parameters
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence< TDoMethods >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_interval< TAxisInterval >::value), "Internal Error: wrong type");

        // define the from and to level indexes
        typedef typename interval_from_index< TAxisInterval >::type FromIndex;
        typedef typename interval_to_index< TAxisInterval >::type ToIndex;

        // compute a set holding all from level indexes
        typedef typename boost::mpl::fold< TDoMethods,
            boost::mpl::set0<>,
            from_indexes_add_functor_do_methods< boost::mpl::_1, boost::mpl::_2 > >::type FromIndexes;

        // compute an ordered vector containing all from levels
        typedef typename boost::mpl::fold< typename make_range< FromIndex, ToIndex >::type,
            boost::mpl::vector0<>,
            boost::mpl::if_< boost::mpl::has_key< FromIndexes, boost::mpl::_2 >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type OrderedFromIndexes;

        // check that there are at least two levels
        GRIDTOOLS_STATIC_ASSERT(
            boost::mpl::size< OrderedFromIndexes >::value >= 2, "there must be at least two levels");

        // iterate over all levels and group succeeding levels into intervals
        // (note that the prior is ok as do methods do not end at the maximum or minimum offsets of a splitter)
        typedef typename boost::mpl::fold<
            boost::mpl::range_c< uint_t, 0, boost::mpl::size< OrderedFromIndexes >::value - 1 >,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                boost::mpl::pair< boost::mpl::at< OrderedFromIndexes, boost::mpl::_2 >,
                                       boost::mpl::prior< boost::mpl::at< OrderedFromIndexes,
                                           boost::mpl::next< boost::mpl::_2 > > > > > >::type type;
    };

} // namespace gridtools
