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
#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/reverse.hpp>

#include "../common/gt_assert.hpp"
#include "amss_descriptor.hpp"
#include "esf_metafunctions.hpp"
#include "grid_traits_metafunctions.hpp"
#include "linearize_mss_functions.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "wrap_type.hpp"
#include "computation_grammar.hpp"

/** @file This file implements the metafunctions to perform data dependency analysis on a
    multi-stage computation (MSS). The idea is to assign to each placeholder used in the
    computation an extent that represents the values that need to be accessed by the stages
    of the computation in each iteration point. This "assignment" is done by using an
    mpl::map between placeholders and extents.
 */

namespace gridtools {

    /**substituting the std::vector type in the args<> with a correspondent
       expandable_parameter placeholder*/
    template < uint_t Size >
    struct substitute_expandable_param {

        template < typename Placeholder >
        struct apply {
            typedef Placeholder type;
        };

        template < ushort_t ID, typename DataStoreType, typename Location, bool Temporary >
        struct apply< arg< ID, std::vector< DataStoreType >, Location, Temporary > > {
            typedef arg< ID, data_store_field< DataStoreType, Size >, Location, Temporary > type;
        };
    };

    /** metafunction removing global accessors from an mpl_vector of pairs <extent, placeholders>.
        Note: the global accessors do not have extents (have mpl::void_ instead). */
    template < typename PlaceholderExtentPair >
    struct remove_global_accessors {
        typedef typename boost::mpl::fold< PlaceholderExtentPair,
            boost::mpl::vector0<>,
            boost::mpl::if_< is_extent< boost::mpl::second< boost::mpl::_2 > >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type type;
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
            GRIDTOOLS_STATIC_ASSERT(
                (is_mss_descriptor< MssDescriptor >::value or MssDescriptor::is_reduction_t::value), GT_INTERNAL_ERROR);

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

                    GRIDTOOLS_STATIC_ASSERT((is_extent< CurrentRange >::value), GT_INTERNAL_ERROR);
                    GRIDTOOLS_STATIC_ASSERT((is_extent< typename PlcRangePair::second >::value), GT_INTERNAL_ERROR);

                    typedef typename sum_extent< CurrentRange, typename PlcRangePair::second >::type candidate_extent;
                    typedef typename enclosing_extent< candidate_extent,
                        typename boost::mpl::at< CurrentMap, typename PlcRangePair::first >::type >::type extent;
                    typedef typename boost::mpl::erase_key< CurrentMap, typename PlcRangePair::first >::type map_erased;
                    typedef typename boost::mpl::insert< map_erased,
                        boost::mpl::pair< typename PlcRangePair::first, extent > >::type type; // new map
                };
            };

            /** metafunction (split into two specializations) to check if a type is an mpl::pair of <placeholder,extent>
             */
            template < typename T >
            struct pair_arg_extent : boost::false_type {};

            template < typename X, typename Y >
            struct pair_arg_extent< boost::mpl::pair< X, Y > > {
                static const bool value = is_arg< X >::value && is_extent< Y >::value;
                typedef boost::mpl::bool_< value > type;
            };

            /** Now we need to update the extent of a given output given the ones of the inputs.
             */
            template < typename Output, typename Inputs, typename CurrentMap >
            struct for_each_output {
                typedef typename boost::mpl::at< CurrentMap, typename Output::first >::type current_extent;

                GRIDTOOLS_STATIC_ASSERT((is_extent< current_extent >::value), GT_INTERNAL_ERROR);

                typedef typename boost::mpl::fold< Inputs,
                    CurrentMap,
                    typename work_on< current_extent >::template with< boost::mpl::_2, boost::mpl::_1 > >::type
                    type; // the new map
            };

            /** Compute the minimum enclosing extents of the list of
                extents provided

                \tparam Extents Sequence of extents
            */
            template < typename Extents >
            struct min_enclosing_extents_of_outputs {
                typedef typename boost::mpl::fold< Extents,
                    extent<>,
                    enclosing_extent< boost::mpl::_1, boost::mpl::_2 > >::type type;
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
            template < typename NewExtent, typename Outputs, typename OriginalMap >
            struct update_map_for_multiple_outputs {
                template < typename TheMap, typename ThePair >
                struct update_value {
                    GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Outputs, pair_arg_extent >::value), GT_INTERNAL_ERROR);

                    // Erasure is needed - we know the key is there otherwise an error would have been catched earlier
                    typedef typename boost::mpl::erase_key< TheMap, typename ThePair::first >::type _Map;
                    typedef typename boost::mpl::insert< _Map,
                        boost::mpl::pair< typename ThePair::first, NewExtent > >::type type;
                };

                typedef typename boost::mpl::fold< Outputs,
                    OriginalMap,
                    update_value< boost::mpl::_1, boost::mpl::_2 > >::type type;
            };

            /**
               From the pairs <placeholders, extents> we need to
               extract the extents corresponding to placeholders in
               the map.

               \tparam Map The map with the extents to extract

               \tparam OutputPairs the sequence of mpl::pairs from which to
               extract the keys to search in the map
            */
            template < typename Map, typename OutputPairs >
            struct extract_output_extents {

                GRIDTOOLS_STATIC_ASSERT((is_sequence_of< OutputPairs, pair_arg_extent >::value), GT_INTERNAL_ERROR);

                template < typename ThePair >
                struct _find_from_second {
                    typedef typename boost::mpl::at< Map, typename ThePair::first >::type type;
                };

                typedef typename boost::mpl::fold< OutputPairs,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back< boost::mpl::_1, _find_from_second< boost::mpl::_2 > > >::type type;
            };

            /** Update map recursively visit the ESFs to process their inputs and outputs
             */
            template < typename ESFs, typename CurrentMap, int Elements >
            struct update_map {
                GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ESFs, is_esf_descriptor >::value), GT_INTERNAL_ERROR);
                typedef typename boost::mpl::at_c< ESFs, 0 >::type current_ESF;
                typedef typename boost::mpl::pop_front< ESFs >::type rest_of_ESFs;

                // First determine which are the outputs
                typedef typename esf_get_w_per_functor< current_ESF, boost::true_type >::type outputs_original;
                GRIDTOOLS_STATIC_ASSERT(
                    (MssDescriptor::is_reduction_t::value || boost::mpl::size< outputs_original >::value),
                    "there seems to be a functor without output fields "
                    "check that each stage has at least one accessor "
                    "defined as \'inout\'");
                typedef typename remove_global_accessors< outputs_original >::type outputs;

#ifndef __CUDACC__
#ifndef ALLOW_EMPTY_EXTENTS
                GRIDTOOLS_STATIC_ASSERT((check_all_extents_are_same_upto< outputs, extent<>, 4 >::type::value),
                    "Horizontal extents of the outputs of ESFs are not all empty. "
                    "All outputs must have empty (horizontal) extents");
#endif
#endif
                GRIDTOOLS_STATIC_ASSERT((is_sequence_of< outputs, pair_arg_extent >::value), GT_INTERNAL_ERROR);

                // We need to check the map here: if the outputs of a
                // single function has different extents in the map we
                // need to update the map with the minimum enclosig
                // extents of those, so that all the subsequent
                // dependencies could be satisfied.

                // First we need to extract the output extents from
                // the map
                typedef typename extract_output_extents< CurrentMap, outputs >::type out_extents;

                // Now we need to get the new extent to be put in the map
                typedef typename min_enclosing_extents_of_outputs< out_extents >::type mee_outputs;

                // Now update the map with the new outputs extents
                typedef
                    typename update_map_for_multiple_outputs< mee_outputs, outputs, CurrentMap >::type NewCurrentMap;

                // mpl::lambda used by the next mpl::tranform
                template < typename NewExtent >
                struct substitute_extent {
                    template < typename Pair >
                    struct apply {
                        typedef typename boost::mpl::pair< typename Pair::first, NewExtent > type;
                    };
                };

                // Now the outputs themselves need to get updated before the next map update
                typedef
                    typename boost::mpl::transform< outputs, substitute_extent< mee_outputs > >::type updated_outputs;

                // Then determine the inputs
                typedef typename esf_get_r_per_functor< current_ESF, boost::true_type >::type inputs;

                GRIDTOOLS_STATIC_ASSERT((is_sequence_of< inputs, pair_arg_extent >::value), GT_INTERNAL_ERROR);

                // Finally, for each output we need to update its
                // extent based on the extents at which the inputs are
                // needed. This makes sense since we are going in
                // reverse orders, from the last to the first stage
                // (esf).
                typedef typename boost::mpl::fold< updated_outputs,
                    NewCurrentMap,
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

    /** This metafunction performs the data-dependence analysis from
        the array of MSSes and a map between placeholders and extents
        initialized, typically with empty extents.  It is taken as
        input since the map values depends on the grid_traits. It
        could be computed here but we would need to pass the
        grid_traits in.

        \tparam MssDescriptorArray The meta-array of MSSes
        \tparam GridTraits The traits of the grids
        \tparam Placeholders The placeholders used in the computation
     */

    template < typename MssDescriptors, typename GridTraits, typename Placeholders >
    struct placeholder_to_extent_map {
      private:
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssDescriptors, is_computation_token >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid_traits_from_id< GridTraits >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Placeholders, is_arg >::value), GT_INTERNAL_ERROR);

        using mss_compute_extent_sizes_t = typename GridTraits::select_mss_compute_extent_sizes;
        using initial_map_of_placeholders_t =
            typename GridTraits::template select_init_map_of_extents< Placeholders >::type;
        // This is where the data-dependence analysis happens
        template < typename CurrentMap, typename Mss >
        using update_extent_map = typename mss_compute_extent_sizes_t::template apply< CurrentMap, Mss >;

      public:
        // we need to iterate over the multistage computations in the computation and
        // update the map accordingly.
        using type = typename boost::mpl::fold< MssDescriptors,
            initial_map_of_placeholders_t,
            update_extent_map< boost::mpl::_1, boost::mpl::_2 > >::type;
    };

    namespace _impl {
        // this is just used to check that the all the outputs of the ESF have the same extent,
        // otherwise the assignment of extents to functors would be not well defined
        template < typename MoP, typename Outputs, typename Extent >
        struct _check_extents_on_outputs {

            template < typename Status, typename Elem >
            struct _new_value {
                typedef typename boost::is_same< typename boost::mpl::at< MoP, Elem >::type, Extent >::type value_t;
                typedef typename boost::mpl::and_< Status, value_t >::type type;
            };

            typedef typename boost::mpl::fold< Outputs,
                boost::true_type,
                _new_value< boost::mpl::_1, boost::mpl::_2 > >::type type;
            static const bool value = type::value;
        };

        template < typename Element >
        struct is_extent_map_element {
            typedef typename is_arg< typename Element::first >::type one;
            typedef typename is_extent< typename Element::second >::type two;

            typedef typename boost::mpl::and_< one, two >::type type;
        };

        template < typename T >
        using is_extent_map = is_sequence_of< T, is_extent_map_element >;

    } // namespace _impl

    template < typename Esf, typename ExtentMap, class = void >
    struct get_extent_for {

        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((_impl::is_extent_map< ExtentMap >::value), GT_INTERNAL_ERROR);

        typedef typename esf_get_w_per_functor< Esf >::type w_plcs;
        typedef typename boost::mpl::at_c< w_plcs, 0 >::type first_out;
        typedef typename boost::mpl::at< ExtentMap, first_out >::type extent;
        // TODO recover
        //                GRIDTOOLS_STATIC_ASSERT((_impl::_check_extents_on_outputs< MapOfPlaceholders, w_plcs,
        //                extent >::value),
        //                    "The output of the ESF do not have all the save extents, so it is not possible to
        //                    select the "
        //                    "extent for the whole ESF.");

        typedef extent type;
    };

    template < typename Esf, typename ExtentMap >
    struct get_extent_for< Esf, ExtentMap, typename std::enable_if< is_esf_with_extent< Esf >::value >::type > {
        using type = typename esf_extent< Esf >::type;
    };

    template < typename Esf, typename ExtentMap >
    struct reduction_get_extent_for {
        typedef typename esf_args< Esf >::type w_plcs;
        typedef typename boost::mpl::at_c< w_plcs, 0 >::type first_out;
        typedef typename boost::mpl::at< ExtentMap, first_out >::type extent;

        // TODO recover
        //                GRIDTOOLS_STATIC_ASSERT((_impl::_check_extents_on_outputs< MapOfPlaceholders, w_plcs,
        //                extent >::value),
        //                    "The output of the ESF do not have all the save extents, so it is not possible to
        //                    select the "
        //                    "extent for the whole ESF.");

        typedef extent type;
    };

    /** This is the metafucntion to iterate over the esfs of a multi-stage stencil
        and gather the outputs (check that they have the same extents), and associate
        to them the corresponding extent */
    template < typename Mss, typename ExtentMap >
    struct get_extent_sizes {
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< Mss >::value), GT_INTERNAL_ERROR);

        // Iterate over each ESF of the MSS to generate a vector of extens whose elements correspond to the elements of
        // the esfs vector (this is to comply with the API for the backend)
        using type = typename boost::mpl::transform< typename mss_descriptor_linear_esf_sequence< Mss >::type,
            get_extent_for< boost::mpl::_, ExtentMap > >::type;
    };

    /** A reduction is always the last stage, so the outputs
        always have the null-extent. In this case then, the map do
        not need to be updated.
     */
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence, typename ExtentMap >
    struct get_extent_sizes< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence >, ExtentMap > {
        // Iterate over each ESF of the REDUCTION
        using type = typename boost::mpl::transform< EsfDescrSequence,
            reduction_get_extent_for< boost::mpl::_, ExtentMap > >::type;
    };

} // namespace gridtools
