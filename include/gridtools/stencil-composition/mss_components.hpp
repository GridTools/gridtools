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
#include "compute_extents_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "make_loop_intervals.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"
#include "stages_maker.hpp"

namespace gridtools {

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     * @tparam RepeatFunctor the length of the chunks for expandable parameters
     */
    template <typename MssDescriptor, typename ExtentMap, typename Axis>
    struct mss_components {
        GT_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), GT_INTERNAL_ERROR);
        typedef MssDescriptor mss_descriptor_t;

        typedef typename mss_descriptor_execution_engine<MssDescriptor>::type execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        typedef typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type linear_esf_t;

        typedef typename get_extent_sizes<MssDescriptor, ExtentMap>::type extent_sizes_t;
        GT_STATIC_ASSERT((is_sequence_of<extent_sizes_t, is_extent>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(
            boost::mpl::size<extent_sizes_t>::value == boost::mpl::size<linear_esf_t>::value, GT_INTERNAL_ERROR);
        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;

        // For historical reasons the user provided axis interval is stripped by one level from the right to produce
        // the interval that will be used for actual computation.
        // TODO(anstaf): fix this ugly convention
        using default_interval_t = interval<typename Axis::FromLevel,
            GT_META_CALL(index_to_level, typename level_to_index<typename Axis::ToLevel>::prior)>;

        // calculate loop intervals and order them according to the execution policy.
        using loop_intervals_t = GT_META_CALL(order_loop_intervals,
            (execution_engine_t,
                GT_META_CALL(make_loop_intervals,
                    (stages_maker<MssDescriptor, ExtentMap>::template apply, default_interval_t))));
    };

    template <typename T>
    GT_META_DEFINE_ALIAS(is_mss_components, meta::is_instantiation_of, (mss_components, T));
} // namespace gridtools
