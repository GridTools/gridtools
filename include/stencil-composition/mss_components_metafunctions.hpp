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

#include "../common/meta_array.hpp"
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

    // TODOCOSUNA unittest this
    /**
     * @brief metafunction that takes an MSS with multiple ESFs and split it into multiple MSS with one ESF each
     * Only to be used for CPU. GPU always fuses ESFs and there is no clear way to split the caches.
     * @tparam MssArray meta array of MSS
     */
    template < typename MssArray >
    struct split_mss_into_independent_esfs {
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of< MssArray, is_computation_token >::value), GT_INTERNAL_ERROR);

        template < typename MssDescriptor >
        struct mss_split_esfs {
            GRIDTOOLS_STATIC_ASSERT((is_computation_token< MssDescriptor >::value), GT_INTERNAL_ERROR);

            typedef typename mss_descriptor_execution_engine< MssDescriptor >::type execution_engine_t;

            template < typename Esf_ >
            struct compose_mss_ {
                typedef mss_descriptor< execution_engine_t, boost::mpl::vector1< Esf_ > > type;
            };

            struct mss_split_multiple_esf {
                typedef typename boost::mpl::fold< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back< boost::mpl::_1, compose_mss_< boost::mpl::_2 > > >::type type;
            };

            typedef typename boost::mpl::if_c<
                // if the number of esf contained in the mss is 1, there is no need to split
                (boost::mpl::size< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type >::value == 1),
                boost::mpl::vector1< MssDescriptor >,
                typename mss_split_multiple_esf::type >::type type;
        };

        typedef meta_array<
            typename boost::mpl::reverse_fold< typename MssArray::elements,
                boost::mpl::vector0<>,
                boost::mpl::copy< boost::mpl::_1, boost::mpl::back_inserter< mss_split_esfs< boost::mpl::_2 > > > >::
                type,
            boost::mpl::quote1< is_computation_token > > type;
    };

    /**
     * @brief metafunction that builds the array of mss components
     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptorArray meta array of mss descriptors
     * @tparam extent_sizes sequence of sequence of extents
     */
    template < enumtype::platform BackendId,
        typename MssDescriptorArray,
        typename ExtentSizes,
        typename RepeatFunctor,
        typename Axis >
    struct build_mss_components_array {
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_computation_token >::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename MssDescriptorArray::elements >::value ==
                                    boost::mpl::size< ExtentSizes >::value),
            GT_INTERNAL_ERROR);

        template < typename _ExtentSizes_ >
        struct unroll_extent_sizes {
            template < typename State, typename Sequence >
            struct insert_unfold {
                typedef typename boost::mpl::fold< Sequence,
                    State,
                    boost::mpl::push_back< boost::mpl::_1, boost::mpl::vector1< boost::mpl::_2 > > >::type type;
            };

            typedef typename boost::mpl::fold< _ExtentSizes_,
                boost::mpl::vector0<>,
                insert_unfold< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            boost::mpl::identity< MssDescriptorArray >,
            split_mss_into_independent_esfs< MssDescriptorArray > >::type mss_array_t;

        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            boost::mpl::identity< ExtentSizes >,
            unroll_extent_sizes< ExtentSizes > >::type extent_sizes_unrolled_t;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< typename mss_array_t::elements >::value ==
                boost::mpl::size< extent_sizes_unrolled_t >::value),
            GT_INTERNAL_ERROR_MSG(
                "Wrong size of the arg_list vector defined inside at least one of the user functions"));

        typedef meta_array< typename boost::mpl::fold<
                                boost::mpl::range_c< int, 0, boost::mpl::size< extent_sizes_unrolled_t >::value >,
                                boost::mpl::vector0<>,
                                boost::mpl::push_back< boost::mpl::_1,
                                    mss_components< boost::mpl::at< typename mss_array_t::elements, boost::mpl::_2 >,
                                                           boost::mpl::at< extent_sizes_unrolled_t, boost::mpl::_2 >,
                                                           RepeatFunctor,
                                                           Axis > > >::type,
            boost::mpl::quote1< is_mss_components > > type;
    }; // struct build_mss_components_array

    /**
     * @brief metafunction that builds a pair of arrays of mss components, to be handled at runtime
     via conditional switches

     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptorArray1 meta array of mss descriptors
     * @tparam MssDescriptorArray2 meta array of mss descriptors
     * @tparam extent_sizes sequence of sequence of extents
     */
    template < enumtype::platform BackendId,
        typename MssDescriptorArray1,
        typename MssDescriptorArray2,
        typename Predicate,
        typename Condition,
        typename ExtentSizes1,
        typename ExtentSizes2,
        typename RepeatFunctor,
        typename Axis >
    struct build_mss_components_array< BackendId,
        meta_array< condition< MssDescriptorArray1, MssDescriptorArray2, Condition >, Predicate >,
        condition< ExtentSizes1, ExtentSizes2, Condition >,
        RepeatFunctor,
        Axis > {
        // typedef typename pair<
        //     typename build_mss_components_array<BackendId, MssDescriptorArray1, ExtentSizes>::type
        //     , typename build_mss_components_array<BackendId, MssDescriptorArray1, ExtentSizes>::type >
        // ::type type;
        typedef condition< typename build_mss_components_array< BackendId,
                               meta_array< MssDescriptorArray1, Predicate >,
                               ExtentSizes1,
                               RepeatFunctor,
                               Axis >::type,
            typename build_mss_components_array< BackendId,
                               meta_array< MssDescriptorArray2, Predicate >,
                               ExtentSizes2,
                               RepeatFunctor,
                               Axis >::type,
            Condition > type;
    }; // build_mss_components_array

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
     * @brief metafunction class that replaces the storage info ID contained in all the
     * placeholders of all temporaries used in Caches. This is needed because the ID is replaced
     * in the aggregator and in order to be able to map the args contained in the aggregator to the
     * args contained in the Cache types we have to replace them in the same way.
     */
    template < typename AggregatorType >
    struct fix_cache_sequence {
        template < typename T >
        struct apply;

        /**
         * @brief specialization for cache_type
         */
        template < template < cache_type, typename, cache_io_policy, typename > class Cache,
            cache_type CacheKind,
            typename Arg,
            cache_io_policy CacheStrategy,
            typename Interval >
        struct apply< Cache< CacheKind, Arg, CacheStrategy, Interval > > {
            GRIDTOOLS_STATIC_ASSERT((is_cache< Cache< CacheKind, Arg, CacheStrategy, Interval > >::value),
                GT_INTERNAL_ERROR_MSG("Given type is no cache."));
            typedef typename boost::mpl::if_< is_tmp_arg< Arg >,
                typename _impl::replace_arg_storage_info< typename AggregatorType::tmp_storage_info_id_t, Arg >::type,
                Arg >::type new_arg_t;
            typedef Cache< CacheKind, new_arg_t, CacheStrategy, Interval > type;
        };

        /**
         * @brief specialization for independent ESF descriptor
         */
        template < template < typename > class IndependentEsfDescriptor, typename ESFVector >
        struct apply< IndependentEsfDescriptor< ESFVector > > {
            typedef typename boost::mpl::transform< ESFVector, fix_cache_sequence >::type fixed_cache_sequence_t;
            typedef IndependentEsfDescriptor< fixed_cache_sequence_t > type;
        };
    };

    /**
     * @brief metafunction class that replaces the storage info ID contained in all the ESF
     * placeholders of all temporaries. This is needed because the ID is replaced in the
     * aggregator and in order to be able to map the args contained in the aggregator to the
     * args contained in the ESF types we have to replace them in the same way.
     */
    template < typename AggregatorType, uint_t RepeatFunctor >
    struct fix_esf_sequence {

        template < typename ArgArray >
        struct impl {
            typedef typename boost::mpl::fold<
                ArgArray,
                boost::mpl::vector0<>,
                boost::mpl::push_back<
                    boost::mpl::_1,
                    boost::mpl::if_< is_tmp_arg< boost::mpl::_2 >,
                        _impl::replace_arg_storage_info< typename AggregatorType::tmp_storage_info_id_t,
                                         boost::mpl::_2 >,
                        boost::mpl::_2 > > >::type new_arg_array_t;
            typedef typename boost::mpl::transform< new_arg_array_t,
                substitute_expandable_param< RepeatFunctor > >::type type;
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

    /**
     * @brief metafunction that replaces the storage info ID contained in all the ESF
     * placeholders of all temporaries. This metafunction is taking an MSS descriptor
     * and iterates (and modifies) all the ESFs.
     */
    template < typename MssDesc, typename Functor >
    struct fix_arg_sequences {
        typedef typename boost::mpl::fold< MssDesc,
            boost::mpl::vector<>,
            boost::mpl::push_back< boost::mpl::_1, fix_arg_sequences< boost::mpl::_2, Functor > > >::type type;
    };

    /**
     * @brief specialization for mss_descriptor types
     */
    template < typename ExecutionEngine, typename ESFSeq, typename CacheSeq, typename Functor >
    struct fix_arg_sequences< mss_descriptor< ExecutionEngine, ESFSeq, CacheSeq >, Functor > {
        typedef mss_descriptor< ExecutionEngine, ESFSeq, CacheSeq > MssDesc;
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< MssDesc >::value), "Given type is no mss_descriptor.");
        typedef typename boost::mpl::transform< ESFSeq, Functor >::type new_esf_sequence_t;
        typedef mss_descriptor< ExecutionEngine, new_esf_sequence_t, CacheSeq > type;
    };

    /**
     * @brief specialization for reduction_descriptor types
     */
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence, typename Functor >
    struct fix_arg_sequences< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence >, Functor > {
        typedef reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > MssDesc;
        GRIDTOOLS_STATIC_ASSERT(
            (is_reduction_descriptor< MssDesc >::value), GT_INTERNAL_ERROR_MSG("Given type is no mss_descriptor."));
        typedef typename boost::mpl::transform< EsfDescrSequence, Functor >::type new_esf_sequence_t;
        typedef reduction_descriptor< ReductionType, BinOp, new_esf_sequence_t > type;
    };

    /**
     * @brief specialization for condition types
     */
    template < typename Sequence1, typename Sequence2, typename Tag, typename Functor >
    struct fix_arg_sequences< condition< Sequence1, Sequence2, Tag >, Functor > {
        typedef condition< Sequence1, Sequence2, Tag > MssDesc;
        GRIDTOOLS_STATIC_ASSERT(
            (is_condition< MssDesc >::value), GT_INTERNAL_ERROR_MSG("Given type is no mss_descriptor."));
        typedef typename fix_arg_sequences< Sequence1, Functor >::type sequence1_t;
        typedef typename fix_arg_sequences< Sequence2, Functor >::type sequence2_t;
        typedef condition< sequence1_t, sequence2_t, Tag > type;
    };

    /**
     * @brief metafunction that replaces the storage info ID contained in all the
     * placeholders of all temporaries. This metafunction is taking an MSS descriptor
     * and iterates (and modifies) all the Caches.
     */
    template < typename MssDesc, typename Functor >
    struct fix_cache_sequences {
        typedef typename boost::mpl::fold< MssDesc,
            boost::mpl::vector<>,
            boost::mpl::push_back< boost::mpl::_1, fix_cache_sequences< boost::mpl::_2, Functor > > >::type type;
    };

    /**
     * @brief specialization for reduction_descriptor types
     */
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence, typename Functor >
    struct fix_cache_sequences< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence >, Functor > {
        typedef reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > type;
    };

    /**
     * @brief specialization for mss_descriptor types
     */
    template < typename ExecutionEngine, typename ESFSeq, typename CacheSeq, typename Functor >
    struct fix_cache_sequences< mss_descriptor< ExecutionEngine, ESFSeq, CacheSeq >, Functor > {
        typedef mss_descriptor< ExecutionEngine, ESFSeq, CacheSeq > MssDesc;
        GRIDTOOLS_STATIC_ASSERT(
            (is_mss_descriptor< MssDesc >::value), GT_INTERNAL_ERROR_MSG("Given type is no mss_descriptor."));
        typedef typename boost::mpl::transform< CacheSeq, Functor >::type new_cache_sequence_t;
        typedef mss_descriptor< ExecutionEngine, ESFSeq, new_cache_sequence_t > type;
    };

    /**
     * @brief specialization for condition types
     */
    template < typename Sequence1, typename Sequence2, typename Tag, typename Functor >
    struct fix_cache_sequences< condition< Sequence1, Sequence2, Tag >, Functor > {
        typedef condition< Sequence1, Sequence2, Tag > MssDesc;
        GRIDTOOLS_STATIC_ASSERT(
            (is_condition< MssDesc >::value), GT_INTERNAL_ERROR_MSG("Given type is no mss_descriptor."));
        typedef typename fix_cache_sequences< Sequence1, Functor >::type sequence1_t;
        typedef typename fix_cache_sequences< Sequence2, Functor >::type sequence2_t;
        typedef condition< sequence1_t, sequence2_t, Tag > type;
    };

    /**
     * @brief metafunction that fixes the storage info type IDs that is contained in
     * all temporary placeholders used in ESF types and Cache types.
     */
    template < typename MetaArray, typename AggregatorType, uint_t RepeatFunctor >
    struct fix_mss_arg_indices;

    template < typename Sequence, typename TPred, typename AggregatorType, uint_t RepeatFunctor >
    struct fix_mss_arg_indices< meta_array< Sequence, TPred >, AggregatorType, RepeatFunctor > {
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);
        typedef
            typename fix_arg_sequences< Sequence, fix_esf_sequence< AggregatorType, RepeatFunctor > >::type tmp_mss_t;
        typedef typename fix_cache_sequences< tmp_mss_t, fix_cache_sequence< AggregatorType > >::type mss_t;
        typedef meta_array< mss_t, TPred > type;
    };

} // namespace gridtools
