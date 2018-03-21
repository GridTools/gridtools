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

#include <boost/mpl/assert.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include "hasdo.hpp"
#include "level.hpp"
#include "interval.hpp"
#include "functor_decorator.hpp"

namespace gridtools {

    /**
     * @struct find_do_method_starting_at
     * Meta function searching for the do method starting at a given from index
     */
    template < typename TFromIndex, typename TToIndex, typename TFunctor >
    struct find_do_method_starting_at {
        // iterate over all do methods starting from a given level and add them to the do method vector
        // (note that the do method vector stores level index pairs instead of intervals)
        typedef typename boost::mpl::fold<
            typename make_range< TFromIndex, TToIndex >::type,
            boost::mpl::vector0<>,
            boost::mpl::if_< has_do< TFunctor, make_interval< TFromIndex, boost::mpl::_2 > >,
                boost::mpl::push_back< boost::mpl::_1, boost::mpl::pair< TFromIndex, boost::mpl::_2 > >,
                boost::mpl::_1 > >::type DoMethods;

        GRIDTOOLS_STATIC_ASSERT(sfinae::has_two_args< TFunctor >::type::value,
            "A functor's Do method is found to have only one argument, when it is supposed to have two");

        // check that:
        // * the k intervals you specified are consistent (i.e. the domain axis used to build
        //     the coordinate system contains all the intervals specified for the solutions)
        // * there is exactly one Do method per functor matching the specified interval
        BOOST_MPL_ASSERT_MSG((boost::mpl::size< DoMethods >::value == 1),
            DID_NOT_FIND_DO_METHOD_FOR_A_GIVEN_INTERVAL_FROM_LEVEL,
            (TFromIndex, DoMethods));

        // define the do method
        typedef typename boost::mpl::back< DoMethods >::type DoMethod;

        // define the do method level offsets
        typedef typename index_to_level< typename boost::mpl::first< DoMethod >::type >::type::Offset FromOffset;
        typedef typename index_to_level< typename boost::mpl::second< DoMethod >::type >::type::Offset ToOffset;

        // check the do method from and to level offsets do not max out the level offset limits
        // (otherwise we cannot guarantee a correct loop level computation afterwards)
        BOOST_MPL_ASSERT_MSG((-cLevelOffsetLimit < FromOffset::value && FromOffset::value < cLevelOffsetLimit &&
                                 -cLevelOffsetLimit < ToOffset::value && ToOffset::value < cLevelOffsetLimit),
            DO_METHOD_DEFINITION_REACHES_LEVEL_OFFSET_LIMIT,
            (TFunctor, DoMethod));

        // return the do method pair holding the from and to indexes of the Do method
        typedef DoMethod type;
    };

    /**
     * @struct are_do_methods_continuous
     * Meta function returning true if two interval pairs are continuous
     */
    template < typename TDoMethod1, typename TDoMethod2 >
    struct are_do_methods_continuous {
        // extract the do method from and to indexes
        typedef typename boost::mpl::second< TDoMethod1 >::type DoMethod1ToIndex;
        typedef typename boost::mpl::first< TDoMethod2 >::type DoMethod2FromIndex;

        // make sure the second interval starts where the first ends
        // (check the index values are continuous and both indexes are associated to the same splitter)
        BOOST_STATIC_CONSTANT(bool,
            value = ((DoMethod1ToIndex::value + 1 == DoMethod2FromIndex::value) &&
                     (index_to_level< DoMethod1ToIndex >::type::Splitter::value ==
                         index_to_level< DoMethod2FromIndex >::type::Splitter::value)));
        typedef boost::mpl::integral_c< bool, value > type;
    };

    /**
     * @struct compute_functor_do_methods
     * Meta function computing a vector containing all do method overloads inside the given axis interval
     * (note that the result vector contains pairs of from and to level indexes instead of intervals)
     */
    template < typename TFunctor, typename TAxisInterval >
    struct compute_functor_do_methods {
        // define the from and to level indexes
        typedef typename interval_from_index< TAxisInterval >::type FromIndex;
        typedef typename interval_to_index< TAxisInterval >::type ToIndex;

        // search all do method from levels
        typedef typename boost::mpl::fold< typename make_range< FromIndex, ToIndex >::type,
            boost::mpl::vector0<>,
            boost::mpl::if_< has_do< TFunctor, index_to_level< boost::mpl::_2 > >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type FromLevels;

        // compute the do method associated to every from level
        typedef typename boost::mpl::transform< FromLevels,
            find_do_method_starting_at< boost::mpl::_, ToIndex, TFunctor > >::type DoMethods;

        // check at least one do method was found
        BOOST_MPL_ASSERT_MSG(
            !boost::mpl::empty< DoMethods >::value, NO_DO_METHOD_FOUND, (TFunctor, TAxisInterval, DoMethods));

        // check the do methods are continuous
        BOOST_MPL_ASSERT_MSG(
            (boost::mpl::fold< boost::mpl::range_c< uint_t, 0, boost::mpl::size< DoMethods >::value - 1 >,
                boost::mpl::true_,
                boost::mpl::if_< are_do_methods_continuous< boost::mpl::at< DoMethods, boost::mpl::_2 >,
                                     boost::mpl::at< DoMethods, boost::mpl::next< boost::mpl::_2 > > >,
                                   boost::mpl::_1,
                                   boost::mpl::false_ > >::type::value),
            DO_METHOD_INTERVALS_ARE_NOT_CONTINOUS,
            (TFunctor, TAxisInterval, DoMethods));

        typedef DoMethods type;
    };

} // namespace gridtools
