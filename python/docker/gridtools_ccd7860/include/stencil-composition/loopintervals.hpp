#pragma once

#include <boost/static_assert.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/prior.hpp>
#include <boost/mpl/if.hpp>
#include "level.hpp"
#include "interval.hpp"

namespace gridtools {
    // implementation of the loop interval computation

    /**
     * @struct from_indexes_add_functor_do_methods
     * Meta function adding all from level indexes of a do method vector to the from index set
     * (note that we introduce an additional from index after the last do method interval)
     */
    template<
        typename TFromIndexSet,
        typename TDoMethods>
    struct from_indexes_add_functor_do_methods
    {
        // check there is at least one do method
        BOOST_MPL_ASSERT_MSG(
                             (!boost::mpl::empty<TDoMethods>::value),
                             NO_FUNCTOR_DO_METHODS_DEFINED_IN_THE_GIVEN_AXIS_INTERVAL,
                             (TDoMethods)
                             );

        // add all do method vector from level indexes
        typedef typename boost::mpl::fold<
            TDoMethods,
            TFromIndexSet,
            boost::mpl::insert<
                boost::mpl::_1,
                boost::mpl::first<boost::mpl::_2>
                >
            >::type NextFromIndexSet;

        // additionally add a sentinel from index after the last do method interval
        // (note that it is ok to use next as we checked the do method intervals to not max out the offset limits)
        typedef typename boost::mpl::insert<
            NextFromIndexSet,
            typename boost::mpl::next<
                typename boost::mpl::second<
                    typename boost::mpl::back<TDoMethods>::type
                    >::type
                >::type
            >::type type;
    };

    /**
     * @struct compute_loop_intervals
     * Meta function computing a vector of loop intervals given a vector of do method vectors and an axis interval
     */
    template<
        typename TDoMethods,
        typename TAxisInterval>
    struct compute_loop_intervals
    {
        // check the template parameters
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence<TDoMethods>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_interval<TAxisInterval>::value), "Internal Error: wrong type");

        // define the from and to level indexes
        typedef typename interval_from_index<TAxisInterval>::type FromIndex;
        typedef typename interval_to_index<TAxisInterval>::type ToIndex;

        // compute a set holding all from level indexes
        typedef typename boost::mpl::fold<
            TDoMethods,
            boost::mpl::set0<>,
            from_indexes_add_functor_do_methods<
                boost::mpl::_1,
                boost::mpl::_2
                >
            >::type FromIndexes;

        // compute an ordered vector containing all from levels
        typedef typename boost::mpl::fold<
            typename make_range<FromIndex, ToIndex>::type,
            boost::mpl::vector0<>,
            boost::mpl::if_<
                boost::mpl::has_key<FromIndexes, boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>,
                boost::mpl::_1
                >
            >::type OrderedFromIndexes;

        // check that there are at least two levels
        GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<OrderedFromIndexes>::value >= 2, "there must be at least two levels");

        // iterate over all levels and group succeeding levels into intervals
        // (note that the prior is ok as do methods do not end at the maximum or minimum offsets of a splitter)
        typedef typename boost::mpl::fold<
            boost::mpl::range_c<uint_t, 0, boost::mpl::size<OrderedFromIndexes>::value - 1>,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::pair<
                    boost::mpl::at<OrderedFromIndexes, boost::mpl::_2>,
                    boost::mpl::prior<
                        boost::mpl::at<
                            OrderedFromIndexes,
                            boost::mpl::next<boost::mpl::_2>
                            >
                        >
                    >
                >
            >::type type;
    };

} // namespace gridtools
