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
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/reverse.hpp>

#include "../common/gt_assert.hpp"
#include "esf_metafunctions.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"

/** @file
    This file implements the metafunctions to perform data dependency analysis on a
    multi-stage computation (MSS). The idea is to assign to each placeholder used in the
    computation an extent that represents the values that need to be accessed by the stages
    of the computation in each iteration point. This "assignment" is done by using an
    mpl::map between placeholders and extents.
 */

namespace gridtools {

    /** \ingroup stancil-composition
     * \{
     */

    /** metafunction removing global accessors from an mpl_vector of pairs <extent, placeholders>.
        Note: the global accessors do not have extents (have mpl::void_ instead). */
    template <typename PlaceholderExtentPair>
    struct remove_global_accessors {
        typedef typename boost::mpl::fold<PlaceholderExtentPair,
            boost::mpl::vector0<>,
            boost::mpl::if_<is_extent<boost::mpl::second<boost::mpl::_2>>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>,
                boost::mpl::_1>>::type type;
    };

    /** This funciton initializes the map between placeholders and extents by
        producing an mpl::map where the keys are the elements of PlaceholdersVector,
        while the values are all InitExtent.

        \tparam PlaceholdersVector vector of placeholders

        \tparam InitExtent Value to associate to each placeholder during the creation of the map
     */
    template <typename PlaceholdersVector, typename InitExtent = extent<>>
    struct init_map_of_extents {
        typedef typename boost::mpl::fold<PlaceholdersVector,
            boost::mpl::map0<>,
            boost::mpl::insert<boost::mpl::_1, boost::mpl::pair<boost::mpl::_2, InitExtent>>>::type type;
    };

    /**
       This is the main entry point for the data dependency
       computation. It starts with an initial map between placeholders
       and extents, which may have been already updated by previous
       compute_extents_of applications to other MSSes in the
       computation. The way of callyng this metafunction is to
       do the following

       \code
       using newmap = compute_extents_of<oldmap>::for_mss<mss>::type;
       \endcode

       \tparam PlaceholdersMap placeholders to extents map from where to start
     */
    template <typename PlaceholdersMap>
    struct compute_extents_of {

        /**
           The for_mss takes the current MSS that needs to be analyzed.

           the `type` is the final map obtained by updating the one provided in
           compute_extents_of

           \tparam MssDescriptor The mulstistage computation to be processed
         */
        template <typename MssDescriptor>
        struct for_mss {
            GT_STATIC_ASSERT(is_mss_descriptor<MssDescriptor>::value, GT_INTERNAL_ERROR);

            /**
               This is the main operation perfromed: we first need to
               extend the extent with the values found in the functor
               arguments, then be sure to put into the map an extent
               that covers the just computed extent with the one
               already present in the map.
             */
            template <typename CurrentRange>
            struct work_on {
                template <typename PlcRangePair, typename CurrentMap>
                struct with {

                    GT_STATIC_ASSERT((is_extent<CurrentRange>::value), GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT((is_extent<typename PlcRangePair::second>::value), GT_INTERNAL_ERROR);

                    typedef typename sum_extent<CurrentRange, typename PlcRangePair::second>::type candidate_extent;
                    using extent = GT_META_CALL(enclosing_extent,
                        (candidate_extent, typename boost::mpl::at<CurrentMap, typename PlcRangePair::first>::type));
                    typedef typename boost::mpl::erase_key<CurrentMap, typename PlcRangePair::first>::type map_erased;
                    typedef typename boost::mpl::insert<map_erased,
                        boost::mpl::pair<typename PlcRangePair::first, extent>>::type type; // new map
                };
            };

            /** metafunction (split into two specializations) to check if a type is an mpl::pair of <placeholder,extent>
             */
            template <typename T>
            struct pair_arg_extent : boost::false_type {};

            template <typename X, typename Y>
            struct pair_arg_extent<boost::mpl::pair<X, Y>> {
                static constexpr bool value = is_plh<X>::value && is_extent<Y>::value;
                typedef boost::mpl::bool_<value> type;
            };

            /** Now we need to update the extent of a given output given the ones of the inputs.
             */
            template <typename Output, typename Inputs, typename CurrentMap>
            struct for_each_output {
                typedef typename boost::mpl::at<CurrentMap, typename Output::first>::type current_extent;

                GT_STATIC_ASSERT((is_extent<current_extent>::value), GT_INTERNAL_ERROR);

                typedef typename boost::mpl::fold<Inputs,
                    CurrentMap,
                    typename work_on<current_extent>::template with<boost::mpl::_2, boost::mpl::_1>>::type
                    type; // the new map
            };

            /** Compute the minimum enclosing extents of the list of
                extents provided

                \tparam Extents Sequence of extents
            */
            template <typename Extents>
            struct min_enclosing_extents_of_outputs {
                typedef typename boost::mpl::
                    fold<Extents, extent<>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type type;
            };

            /**
               Given the map between placeholders and extents, this
               metafunction produce another map in which the
               placeholders in Outputs::first are updated with the
               extent in NewExtent.

               \tparam NewExtent The new extent to insert into the map
               \tparam Outputs Sequence of mpl::pairs of Outputs and extents
               \tparam OrigialMap The map to be updated
             */
            template <typename NewExtent, typename Outputs, typename OriginalMap>
            struct update_map_for_multiple_outputs {
                template <typename TheMap, typename ThePair>
                struct update_value {
                    GT_STATIC_ASSERT((is_sequence_of<Outputs, pair_arg_extent>::value), GT_INTERNAL_ERROR);

                    // Erasure is needed - we know the key is there otherwise an error would have been catched earlier
                    typedef typename boost::mpl::erase_key<TheMap, typename ThePair::first>::type _Map;
                    typedef
                        typename boost::mpl::insert<_Map, boost::mpl::pair<typename ThePair::first, NewExtent>>::type
                            type;
                };

                typedef
                    typename boost::mpl::fold<Outputs, OriginalMap, update_value<boost::mpl::_1, boost::mpl::_2>>::type
                        type;
            };

            /**
               From the pairs <placeholders, extents> we need to
               extract the extents corresponding to placeholders in
               the map.

               \tparam Map The map with the extents to extract

               \tparam OutputPairs the sequence of mpl::pairs from which to
               extract the keys to search in the map
            */
            template <typename Map, typename OutputPairs>
            struct extract_output_extents {

                GT_STATIC_ASSERT((is_sequence_of<OutputPairs, pair_arg_extent>::value), GT_INTERNAL_ERROR);

                template <typename ThePair>
                struct _find_from_second {
                    typedef typename boost::mpl::at<Map, typename ThePair::first>::type type;
                };

                typedef typename boost::mpl::fold<OutputPairs,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back<boost::mpl::_1, _find_from_second<boost::mpl::_2>>>::type type;
            };

            /** Update map recursively visit the ESFs to process their inputs and outputs
             */
            template <typename ESFs, typename CurrentMap, int Elements>
            struct update_map {
                GT_STATIC_ASSERT((is_sequence_of<ESFs, is_esf_descriptor>::value), GT_INTERNAL_ERROR);
                typedef typename boost::mpl::at_c<ESFs, 0>::type current_ESF;
                typedef typename boost::mpl::pop_front<ESFs>::type rest_of_ESFs;

                // First determine which are the outputs
                typedef typename esf_get_w_per_functor<current_ESF, boost::true_type>::type outputs_original;
                GT_STATIC_ASSERT(boost::mpl::size<outputs_original>::value,
                    "there seems to be a functor without output fields "
                    "check that each stage has at least one accessor "
                    "defined as \'inout\'");
                typedef typename remove_global_accessors<outputs_original>::type outputs;

#ifndef __CUDACC__
                GT_STATIC_ASSERT(check_all_horizotal_extents_are_zero<outputs>::type::value,
                    "Horizontal extents of the outputs of ESFs are not all empty. "
                    "All outputs must have empty (horizontal) extents");
#endif
                GT_STATIC_ASSERT((is_sequence_of<outputs, pair_arg_extent>::value), GT_INTERNAL_ERROR);

                // We need to check the map here: if the outputs of a
                // single function has different extents in the map we
                // need to update the map with the minimum enclosig
                // extents of those, so that all the subsequent
                // dependencies could be satisfied.

                // First we need to extract the output extents from
                // the map
                typedef typename extract_output_extents<CurrentMap, outputs>::type out_extents;

                // Now we need to get the new extent to be put in the map
                typedef typename min_enclosing_extents_of_outputs<out_extents>::type mee_outputs;

                // Now update the map with the new outputs extents
                typedef typename update_map_for_multiple_outputs<mee_outputs, outputs, CurrentMap>::type NewCurrentMap;

                // mpl::lambda used by the next mpl::tranform
                template <typename NewExtent>
                struct substitute_extent {
                    template <typename Pair>
                    struct apply {
                        typedef typename boost::mpl::pair<typename Pair::first, NewExtent> type;
                    };
                };

                // Now the outputs themselves need to get updated before the next map update
                typedef typename boost::mpl::transform<outputs, substitute_extent<mee_outputs>>::type updated_outputs;

                // Then determine the inputs
                typedef typename esf_get_r_per_functor<current_ESF, boost::true_type>::type inputs;

                GT_STATIC_ASSERT((is_sequence_of<inputs, pair_arg_extent>::value), GT_INTERNAL_ERROR);

                // Finally, for each output we need to update its
                // extent based on the extents at which the inputs are
                // needed. This makes sense since we are going in
                // reverse orders, from the last to the first stage
                // (esf).
                typedef typename boost::mpl::fold<updated_outputs,
                    NewCurrentMap,
                    for_each_output<boost::mpl::_2, inputs, boost::mpl::_1>>::type new_map;

                typedef
                    typename update_map<rest_of_ESFs, new_map, boost::mpl::size<rest_of_ESFs>::type::value>::type type;
            };

            /** Base of recursion */
            template <typename ESFs, typename CurrentMap>
            struct update_map<ESFs, CurrentMap, 0> {
                typedef CurrentMap type;
            };

            // We need to obtain the proper linearization (unfolding
            // independents) of the list of stages and and then we
            // need to go from the outputs to the inputs (the reverse)
            using ESFs = GT_META_CALL(
                meta::reverse, GT_META_CALL(unwrap_independent, typename MssDescriptor::esf_sequence_t));

            // The return of this metafunction is here. We need to
            // update the map of plcaholderss. A numerical value helps
            // in determining the iteration space
            typedef typename update_map<ESFs, PlaceholdersMap, boost::mpl::size<ESFs>::type::value>::type type;
        }; // struct for_mss
    };     // struct compute_extents_of

    /** This metafunction performs the data-dependence analysis from
        the array of MSSes and a map between placeholders and extents
        initialized, typically with empty extents.

        \tparam MssDescriptorArray The meta-array of MSSes
        \tparam Placeholders The placeholders used in the computation
     */

    template <typename MssDescriptors, typename Placeholders>
    struct placeholder_to_extent_map {
      private:
        GT_STATIC_ASSERT((is_sequence_of<MssDescriptors, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_sequence_of<Placeholders, is_plh>::value), GT_INTERNAL_ERROR);

        // This is where the data-dependence analysis happens
        template <typename PlaceholdersMap, typename Mss>
        struct update_extent_map : compute_extents_of<PlaceholdersMap>::template for_mss<Mss> {};

      public:
        // we need to iterate over the multistage computations in the computation and
        // update the map accordingly.
        using type = typename boost::mpl::fold<MssDescriptors,
            typename init_map_of_extents<Placeholders>::type,
            update_extent_map<boost::mpl::_1, boost::mpl::_2>>::type;
    };

    namespace _impl {
        // this is just used to check that the all the outputs of the ESF have the same extent,
        // otherwise the assignment of extents to functors would be not well defined
        template <typename MoP, typename Outputs, typename Extent>
        struct _check_extents_on_outputs {

            template <typename Status, typename Elem>
            struct _new_value {
                typedef typename boost::is_same<typename boost::mpl::at<MoP, Elem>::type, Extent>::type value_t;
                typedef typename boost::mpl::and_<Status, value_t>::type type;
            };

            typedef
                typename boost::mpl::fold<Outputs, boost::true_type, _new_value<boost::mpl::_1, boost::mpl::_2>>::type
                    type;
            static constexpr bool value = type::value;
        };

        template <typename Element>
        struct is_extent_map_element {
            typedef typename is_plh<typename Element::first>::type one;
            typedef typename is_extent<typename Element::second>::type two;

            typedef typename boost::mpl::and_<one, two>::type type;
        };

        template <typename T>
        using is_extent_map = is_sequence_of<T, is_extent_map_element>;

    } // namespace _impl

    template <typename Esf, typename ExtentMap, class = void>
    struct get_extent_for {

        GT_STATIC_ASSERT(is_esf_descriptor<Esf>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(_impl::is_extent_map<ExtentMap>::value, GT_INTERNAL_ERROR);

        using w_plcs = typename esf_get_w_per_functor<Esf>::type;
        using first_out = typename boost::mpl::at_c<w_plcs, 0>::type;
        using type = typename boost::mpl::at<ExtentMap, first_out>::type;
        // TODO recover
        //                GT_STATIC_ASSERT((_impl::_check_extents_on_outputs< MapOfPlaceholders, w_plcs,
        //                extent >::value),
        //                    "The output of the ESF do not have all the save extents, so it is not possible to
        //                    select the "
        //                    "extent for the whole ESF.");
    };

    template <typename Esf, typename ExtentMap>
    struct get_extent_for<Esf, ExtentMap, enable_if_t<is_esf_with_extent<Esf>::value>> : esf_extent<Esf> {};

    /**
     * @}
     */
} // namespace gridtools
