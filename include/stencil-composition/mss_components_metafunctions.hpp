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
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/zip_view.hpp>

#include "../common/gt_assert.hpp"
#include "./reductions/reduction_descriptor.hpp"
#include "grid.hpp"
#include "mss_components.hpp"
#include "mss_metafunctions.hpp"
#include "functor_decorator.hpp"
#include "sfinae.hpp"

namespace gridtools {

    template < typename T >
    struct mss_components_is_reduction;

    template < typename MssDescriptor, typename ExtentSizes, typename RepeatFunctor, typename Axis >
    struct mss_components_is_reduction< mss_components< MssDescriptor, ExtentSizes, RepeatFunctor, Axis > >
        : MssDescriptor::is_reduction_t {};

    template < typename MssDescriptor >
    struct mss_split_esfs {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token< MssDescriptor >::value), GT_INTERNAL_ERROR);

        using execution_engine_t = typename mss_descriptor_execution_engine< MssDescriptor >::type;

        template < typename Esf_ >
        using compose_mss_ = mss_descriptor< execution_engine_t, boost::mpl::vector1< Esf_ > >;

        using mss_split_multiple_esf_t =
            typename boost::mpl::fold< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1, compose_mss_< boost::mpl::_2 > > >::type;

        using type = typename boost::mpl::if_c<
            // if the number of esf contained in the mss is 1, there is no need to split
            (boost::mpl::size< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type >::value == 1),
            boost::mpl::vector1< MssDescriptor >,
            mss_split_multiple_esf_t >::type;
    };

    // TODOCOSUNA unittest this
    /**
     * @brief metafunction that takes an MSS with multiple ESFs and split it into multiple MSS with one ESF each
     * Only to be used for CPU. GPU always fuses ESFs and there is no clear way to split the caches.
     * @tparam Msses computaion token sequence
     */
    template < typename Msses >
    struct split_mss_into_independent_esfs {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Msses, is_computation_token >::value), GT_INTERNAL_ERROR);

        typedef typename boost::mpl::reverse_fold< Msses,
            boost::mpl::vector0<>,
            boost::mpl::copy< boost::mpl::_1, boost::mpl::back_inserter< mss_split_esfs< boost::mpl::_2 > > > >::type
            type;
    };

    /**
     * @brief metafunction that builds the array of mss components
     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptors mss descriptor sequence
     * @tparam extent_sizes sequence of sequence of extents
     */
    template < typename MssFuseEsfStrategy,
        typename MssDescriptors,
        typename ExtentMap,
        typename RepeatFunctor,
        typename Axis >
    struct build_mss_components_array {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssDescriptors, is_computation_token >::value), GT_INTERNAL_ERROR);

        using mss_seq_t = typename boost::mpl::eval_if< MssFuseEsfStrategy,
            boost::mpl::identity< MssDescriptors >,
            split_mss_into_independent_esfs< MssDescriptors > >::type;

        using type = typename boost::mpl::transform< mss_seq_t,
            mss_components< boost::mpl::_, get_extent_sizes< boost::mpl::_, ExtentMap >, RepeatFunctor, Axis > >::type;
    };

    /**
     * @brief metafunction that computes the mss functor do methods
     */
    template < typename MssComponents, typename Grid >
    struct mss_functor_do_methods {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);

        /**
         *  compute the functor do methods - This is the most computationally intensive part
         */
        template < typename Functor >
        struct inserter_ {

            typedef typename boost::mpl::if_< typename sfinae::has_two_args< Functor >::type,
                Functor,
                functor_default_interval< Functor, typename Grid::axis_type > >::type functor_t;

            typedef typename compute_functor_do_methods< functor_t, typename Grid::axis_type >::type type;
        };

        typedef typename boost::mpl::transform< typename MssComponents::functors_seq_t,
            inserter_< boost::mpl::_ > >::type
            type; // Vector of vectors - each element is a vector of pairs of actual axis-indices
    };

    /**
     * @brief metafunction that computes the loop intervals of an mss
     */
    template < typename MssComponents, typename Grid >
    struct mss_loop_intervals {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

        /**
         *  compute the functor do methods - This is the most computationally intensive part
         */
        typedef typename mss_functor_do_methods< MssComponents, Grid >::type functor_do_methods;

        /**
         * compute the loop intervals
         */
        typedef typename compute_loop_intervals< functor_do_methods,
            typename Grid::axis_type >::type type; // vector of pairs of indices - sorted and contiguous
    };

    template < typename MssComponents, typename Grid >
    struct mss_functor_do_method_lookup_maps {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);
        typedef typename mss_functor_do_methods< MssComponents, Grid >::type functor_do_methods;

        typedef typename mss_loop_intervals< MssComponents, Grid >::type loop_intervals;
        /**
         * compute the do method lookup maps
         *
         */
        typedef typename boost::mpl::transform< functor_do_methods,
            compute_functor_do_method_lookup_map< boost::mpl::_, loop_intervals > >::type
            type; // vector of maps, indexed by functors indices in Functor vector.
    };

    /**
     * @brief metafunction class that replaces the storage info ID contained in all the ESF
     * placeholders of all temporaries. This is needed because the ID is replaced in the
     * aggregator and in order to be able to map the args contained in the aggregator to the
     * args contained in the ESF types we have to replace them in the same way.
     */
    template < uint_t RepeatFunctor >
    struct fix_esf_sequence {

        template < typename ArgArray >
        struct impl {
            typedef typename boost::mpl::transform< ArgArray, substitute_expandable_param< RepeatFunctor > >::type type;
        };

        template < typename T >
        struct apply;

        /**
         * @brief specialization for structured grid ESF types
         */
        template < template < typename, typename, typename > class EsfDescriptor,
            typename ESF,
            typename ArgArray,
            typename Staggering >
        struct apply< EsfDescriptor< ESF, ArgArray, Staggering > > {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfDescriptor< ESF, ArgArray, Staggering > >::value),
                GT_INTERNAL_ERROR_MSG("Given type is no esf_descriptor."));
            typedef EsfDescriptor< ESF, typename impl< ArgArray >::type, Staggering > type;
        };

        /**
         * @brief specialization for structured grid ESF with extent
         */
        template < template < typename, typename, typename, typename > class EsfDescriptor,
            typename ESF,
            typename Extent,
            typename ArgArray,
            typename Staggering >
        struct apply< EsfDescriptor< ESF, Extent, ArgArray, Staggering > > {
            static_assert(is_esf_descriptor< EsfDescriptor< ESF, Extent, ArgArray, Staggering > >::value,
                GT_INTERNAL_ERROR_MSG("Type is not an EsfDescriptor"));
            typedef EsfDescriptor< ESF, Extent, typename impl< ArgArray >::type, Staggering > type;
        };

        /**
         * @brief specialization for icosahedral grid ESF types
         */
        template < template < template < uint_t > class, typename, typename, typename, typename > class EsfDescriptor,
            template < uint_t > class ESF,
            typename Topology,
            typename LocationType,
            typename Color,
            typename ArgArray >
        struct apply< EsfDescriptor< ESF, Topology, LocationType, Color, ArgArray > > {
            GRIDTOOLS_STATIC_ASSERT(
                (is_esf_descriptor< EsfDescriptor< ESF, Topology, LocationType, Color, ArgArray > >::value),
                GT_INTERNAL_ERROR_MSG("Given type is no esf_descriptor."));
            typedef EsfDescriptor< ESF, Topology, LocationType, Color, typename impl< ArgArray >::type > type;
        };

        /**
         * @brief specialization for independent ESF descriptor
         */
        template < template < typename > class IndependentEsfDescriptor, typename ESFVector >
        struct apply< IndependentEsfDescriptor< ESFVector > > {
            typedef typename boost::mpl::transform< ESFVector, fix_esf_sequence >::type fixed_esf_sequence_t;
            typedef IndependentEsfDescriptor< fixed_esf_sequence_t > type;
        };
    };

    template < typename T, typename Functor >
    struct fix_arg_sequences;

    /**
     * @brief specialization for mss_descriptor types
     */
    template < typename ExecutionEngine, typename ESFSeq, typename CacheSeq, typename Functor >
    struct fix_arg_sequences< mss_descriptor< ExecutionEngine, ESFSeq, CacheSeq >, Functor > {
        using type =
            mss_descriptor< ExecutionEngine, typename boost::mpl::transform< ESFSeq, Functor >::type, CacheSeq >;
    };

    /**
     * @brief specialization for reduction_descriptor types
     */
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence, typename Functor >
    struct fix_arg_sequences< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence >, Functor > {
        using type = reduction_descriptor< ReductionType,
            BinOp,
            typename boost::mpl::transform< EsfDescrSequence, Functor >::type >;
    };

    /**
     * @brief metafunction that fixes the storage info type IDs that is contained in
     * all temporary placeholders used in ESF types and Cache types.
     */
    template < typename Sequence, uint_t RepeatFunctor >
    struct fix_mss_arg_indices {
        typedef typename fix_arg_sequences< Sequence, fix_esf_sequence< RepeatFunctor > >::type type;
    };

    template < uint_t RepeatFunctor >
    struct fix_mss_arg_indices_f {
        template < typename T >
        using res_t = typename fix_mss_arg_indices< T, RepeatFunctor >::type;

        template < typename T >
        res_t< T > operator()(T) const {
            return {};
        }
        template < typename R, typename B, typename E >
        res_t< reduction_descriptor< R, B, E > > operator()(reduction_descriptor< R, B, E > src) const {
            return {src.get()};
        }
    };

} // namespace gridtools
