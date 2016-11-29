#pragma once
#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/reverse.hpp>

#include "./amss_descriptor.hpp"
#include "./conditionals/condition.hpp"
#include "./esf_metafunctions.hpp"
#include "./grid_traits_metafunctions.hpp"
#include "./linearize_mss_functions.hpp"
#include "./mss.hpp"
#include "./mss_metafunctions.hpp"
#include "./reductions/reduction_descriptor.hpp"
#include "./wrap_type.hpp"

/** @file This file implements the metafunctions to perform data dependency analysis on a
    multi-stage computation (MSS). The idea is to assign to each placeholder used in the
    computation an extent that represents the values that need to be accessed by the stages
    of the computation in each iteration point. This "assignment" is done by using an
    mpl::map between placeholders and extents.
 */

namespace gridtools {

    template < typename Storage, uint_t >
    struct expandable_parameters;

    /**substituting the std::vector type in the args<> with a correspondent
       expandable_parameter placeholder*/
    template < uint_t Size >
    struct substitute_expandable_param {

        template < typename Placeholder >
        struct apply {
            typedef Placeholder type;
        };

        template < ushort_t ID, typename Storage >
        struct apply< arg< ID, std::vector< pointer< storage< Storage > > > > > {
            typedef arg< ID, storage< expandable_parameters< typename Storage::basic_type, Size > > > type;
        };

        template < ushort_t ID, typename Storage >
        struct apply< arg< ID, std::vector< pointer< no_storage_type_yet< storage< Storage > > > > > > {
            typedef arg< ID,
                no_storage_type_yet< storage< expandable_parameters< typename Storage::basic_type, Size > > > > type;
        };

        template < typename Arg, typename Extent >
        struct apply< boost::mpl::pair< Arg, Extent > > {
            typedef boost::mpl::pair< typename apply< Arg >::type, Extent > type;
        };
    };

    template < typename PlaceholderArray, uint_t Size >
    struct substitute_expandable_params {
        typedef typename boost::mpl::transform< PlaceholderArray, substitute_expandable_param< Size > >::type type;
    };

    /** This funciton initializes the map between placeholders and extents by
        producing an mpl::map where the keys are the elements of PlaceholdersVector,
        while the values are all InitExtent.

        \tparam PlaceholdersVector vector of placeholders

        \tparam InitExtent Value to associate to each placeholder during the creation of the map
     */
    template < typename PlaceholdersVector, typename InitExtent = extent<> >
    struct init_map_of_extents {
        typedef typename boost::mpl::fold< PlaceholdersVector,
            boost::mpl::map0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::pair< boost::mpl::_2, InitExtent > > >::type type;
    };

    /**
       This is the main entry point for the data dependency
       computation. It starts with an initial map between placeholders
       and extents, which may have been already updated by previous
       compute_extents_of applications to other MSSes in the
       computation. The way of callyng this metafunction is to
       do the following

       \begincode
       using newmap = compute_extents_of<oldmap>::for_mss<mss>::type;
       \endcode

       \tparam PlaceholdersMap placeholders to extents map from where to start
     */
    template < typename PlaceholdersMap, uint_t RepeatFunctor >
    struct compute_extents_of {
        /**
           The for_mss takes the current MSS that needs to be analyzed.

#ifdef STRUCTURED_GRIDS
    #include "stencil-composition/structured_grids/compute_extents_metafunctions.hpp"
#else
    #include "stencil-composition/icosahedral_grids/compute_extents_metafunctions.hpp"
#endif
