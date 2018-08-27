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
#include "./linearize_mss_functions.hpp"
#include "computation_grammar.hpp"
#include "compute_extents_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "functor_decorator.hpp"
#include "make_loop_intervals.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "stages_maker.hpp"

namespace gridtools {

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     * @tparam RepeatFunctor the length of the chunks for expandable parameters
     */
    template <typename MssDescriptor, typename ExtentMap, typename RepeatFunctor, typename Axis>
    struct mss_components {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token<MssDescriptor>::value), GT_INTERNAL_ERROR);
        typedef MssDescriptor mss_descriptor_t;

        typedef typename mss_descriptor_execution_engine<MssDescriptor>::type execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        typedef typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type linear_esf_t;

        /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
        typedef typename boost::mpl::transform<linear_esf_t, esf_get_w_temps_per_functor<boost::mpl::_>>::type
            written_temps_per_functor_t;

        /**
         * typename linear_esf is a list of all the esf nodes in the multi-stage descriptor.
         * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
         */
        typedef typename boost::mpl::transform<linear_esf_t, extract_esf_functor>::type functors_seq_t;

        /*
          @brief attaching an integer index to each functor

          This ensures that the types in the functors_list_t are unique.
          It is necessary to have unique types in the functors_list_t, so that we can use the
          functor types as keys in an MPL map. This is used in particular in the innermost loop, where
          we decide at compile-time wether the functors need synchronization or not, based on a map
          connecting the functors to the "is independent" boolean (set to true if the functor does
          not have data dependency with the next one). Since we can have the exact same functor used multiple
          times in an MSS both as dependent or independent, we cannot use the plain functor type as key for the
          above mentioned map, and we need to attach a unique index to its type.
        */
        typedef typename boost::mpl::fold<boost::mpl::range_c<ushort_t, 0, boost::mpl::size<functors_seq_t>::value>,
            boost::mpl::vector0<>,
            boost::mpl::push_back<boost::mpl::_1,
                functor_decorator<boost::mpl::_2,
                    boost::mpl::at<functors_seq_t, boost::mpl::_2>,
                    RepeatFunctor,
                    Axis>>>::type functors_list_t;

        typedef typename get_extent_sizes<MssDescriptor, ExtentMap>::type extent_sizes_t;
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<extent_sizes_t, is_extent>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(
            boost::mpl::size<extent_sizes_t>::value == boost::mpl::size<linear_esf_t>::value, GT_INTERNAL_ERROR);
        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;

        using default_interval_t = interval<typename Axis::FromLevel,
            GT_META_CALL(index_to_level, typename GT_META_CALL(level_to_index, typename Axis::ToLevel)::prior)>;

        using loop_intervals_t = GT_META_CALL(order_loop_intervals,
            (execution_engine_t,
                GT_META_CALL(make_loop_intervals,
                    (stages_maker<MssDescriptor, ExtentMap, RepeatFunctor::value>::template apply,
                        default_interval_t))));
    };

    template <typename T>
    struct is_mss_components : boost::mpl::false_ {};

    template <typename MssDescriptor, typename ExtentMap, typename RepeatFunctor, typename Axis>
    struct is_mss_components<mss_components<MssDescriptor, ExtentMap, RepeatFunctor, Axis>> : boost::mpl::true_ {};

} // namespace gridtools
