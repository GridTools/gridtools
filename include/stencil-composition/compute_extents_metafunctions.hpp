#pragma once
#include <boost/mpl/fold.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>

#include "./esf_metafunctions.hpp"
#include "./wrap_type.hpp"
#include "./mss.hpp"
#include "./amss_descriptor.hpp"
#include "./mss_metafunctions.hpp"
#include "./reductions/reduction_descriptor.hpp"
#include "./linearize_mss_functions.hpp"

/** @file This file implements the metafunctions to perform data dependency analysis on a
    multi-stage computation (MSS). The idea is to assign to each placeholder used in the
    computation an extent that represents the values that need to be accessed by the stages
    of the computation in each iteration point. This "assignment" is done by using an
    mpl::map between placeholders and extents.
 */

namespace gridtools {

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
    template < typename PlaceholdersMap >
    struct compute_extents_of {

        /**
           The for_mss takes the current MSS that needs to be analyzed.

           the ::type is the final map obtained by updating the one provided in
           compute_extents_of

           \tparam MssDescriptor The mulstistage computation to be processed
         */
        template < typename MssDescriptor >
        struct for_mss {
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< MssDescriptor >::value or MssDescriptor::is_reduction_t::value),
                "Internal Error: invalid type");

            /**
               This is the main operation perfromed: we first need to
               extend the extent with the values found in the functor
               arguments, then be sure to put into the map an extent
               that covers the just computed extent with the one
               already present in the map.
             */
            template < typename CurrentRange >
            struct work_on {
                template < typename PlcRangePair, typename CurrentMap >
                struct with {
                    typedef typename sum_extent< CurrentRange, typename PlcRangePair::second >::type candidate_extent;
                    typedef typename enclosing_extent< candidate_extent,
                        typename boost::mpl::at< CurrentMap, typename PlcRangePair::first >::type >::type extent;
                    typedef typename boost::mpl::erase_key< CurrentMap, typename PlcRangePair::first >::type map_erased;
                    typedef typename boost::mpl::insert< map_erased,
                        boost::mpl::pair< typename PlcRangePair::first, extent > >::type type; // new map
                };
            };

            /** Now we need to update the extent of a given output given the ones of the inputs.
             */
            template < typename Output, typename Inputs, typename CurrentMap >
            struct for_each_output {
                typedef typename boost::mpl::at< CurrentMap, typename Output::first >::type current_extent;

                typedef typename boost::mpl::fold< Inputs,
                    CurrentMap,
                    typename work_on< current_extent >::template with< boost::mpl::_2, boost::mpl::_1 > >::type
                    type; // the new map
            };

            /** Update map recursively visit the ESFs to process their inputs and outputs
             */
            template < typename ESFs, typename CurrentMap, int Elements >
            struct update_map {
                typedef typename boost::mpl::at_c< ESFs, 0 >::type current_ESF;
                typedef typename boost::mpl::pop_front< ESFs >::type rest_of_ESFs;

                // First determine which are the outputs
                typedef typename esf_get_w_per_functor< current_ESF, boost::true_type >::type outputs;
                GRIDTOOLS_STATIC_ASSERT((check_all_extents_are< outputs, extent<> >::type::value),
                    "Extents of the outputs of ESFs are not all empty. All outputs must have empty extents");

                // Then determine the inputs
                typedef typename esf_get_r_per_functor< current_ESF, boost::true_type >::type inputs;

                // Finally, for each output we need to update its
                // extent based on the extents at which the inputs are
                // needed. This makes sense since we are going in
                // reverse orders, from the last to the first stage
                // (esf).
                typedef typename boost::mpl::fold< outputs,
                    CurrentMap,
                    for_each_output< boost::mpl::_2, inputs, boost::mpl::_1 > >::type new_map;

                typedef
                    typename update_map< rest_of_ESFs, new_map, boost::mpl::size< rest_of_ESFs >::type::value >::type
                        type;
            };

            /** Base of recursion */
            template < typename ESFs, typename CurrentMap >
            struct update_map< ESFs, CurrentMap, 0 > {
                typedef CurrentMap type;
            };

            // We need to obtain the proper linearization (unfolding
            // independents) of the list of stages and and then we
            // need to go from the outputs to the inputs (the reverse)
            typedef typename boost::mpl::reverse< typename unwrap_independent<
                typename mss_descriptor_esf_sequence< MssDescriptor >::type >::type >::type ESFs;

            // The return of this metafunction is here. We need to
            // update the map of plcaholderss. A numerical value helps
            // in determining the iteration space
            typedef typename update_map< ESFs, PlaceholdersMap, boost::mpl::size< ESFs >::type::value >::type type;
        }; // struct for_mss
    };     // struct compute_extents_of
}
