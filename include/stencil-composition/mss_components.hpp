/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "mss.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "esf_metafunctions.hpp"
#include "mss_metafunctions.hpp"
#include "./linearize_mss_functions.hpp"
#include "functor_decorator.hpp"

namespace gridtools {

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     * @tparam RepeatFunctor the length of the chunks for expandable parameters, see @ref
     * gridtools::expandable_parameters
     */
    template < typename MssDescriptor, typename ExtentSizes, typename RepeatFunctor >
    struct mss_components {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token< MssDescriptor >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< ExtentSizes >::type::value == 0 || is_sequence_of< ExtentSizes, is_extent >::value),
            "Internal Error: wrong type");
        typedef MssDescriptor mss_descriptor_t;

        typedef typename mss_descriptor_execution_engine< MssDescriptor >::type execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        typedef typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type linear_esf_t;

        /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
        typedef typename boost::mpl::fold< linear_esf_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, esf_get_w_temps_per_functor< boost::mpl::_2 > > >::type
            written_temps_per_functor_t;

        /**
         * typename linear_esf is a list of all the esf nodes in the multi-stage descriptor.
         * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
         */
        typedef typename boost::mpl::transform< linear_esf_t, extract_esf_functor >::type functors_seq_t;

        /*
          @brief attaching an integer index to each functor

          This ensures that the types in the functors_list_t are unique.
          It is necessary to have unique types in the functors_list_t, so that we can use the
          functor types as keys in an MPL map. This is used in particular in the innermost loop, where
          we decide at compile-time wether the functors need synchronization or not, based on a map
          connecting the functors to the "is independent" boolean (set to true if the functor does
          not have data dependency with the next one). Since we can have the exact same functor used multiple
          times in an MSS both as dependent or independent, we cannot use the plain functor type as key for the
          abovementioned map, and we need to attach a unique index to its type.
        */
        typedef
            typename boost::mpl::fold< boost::mpl::range_c< ushort_t, 0, boost::mpl::size< functors_seq_t >::value >,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                                           functor_decorator< boost::mpl::_2,
                                               boost::mpl::at< functors_seq_t, boost::mpl::_2 >,
                                               RepeatFunctor > > >::type functors_list_t;

        typedef ExtentSizes extent_sizes_t;
        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;
    };

    template < typename T >
    struct is_mss_components : boost::mpl::false_ {};

    template < typename MssDescriptor, typename ExtentSizes, typename RepeatFunctor >
    struct is_mss_components< mss_components< MssDescriptor, ExtentSizes, RepeatFunctor > > : boost::mpl::true_ {};

} // namespace gridtools
