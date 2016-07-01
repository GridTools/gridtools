/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "mss.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "esf_metafunctions.hpp"
#include "mss_metafunctions.hpp"
#include "./linearize_mss_functions.hpp"

namespace gridtools {

    /**
       @brief MPL pair wrapper with more meaningful type names for the specific use case.
    */
    template < typename T1, typename T2, typename Repeat >
    struct functor_id_pair {
        typedef Repeat repeat_t;
        typedef T1 id;
        typedef T2 f_type;
    };

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     * @tparam RepeatFunctor the length of the chunks for expandable parameters, see @ref gridtools::expandable_parameters
     */
    template < typename MssDescriptor, typename ExtentSizes, typename RepeatFunctor >
    struct mss_components {
        GRIDTOOLS_STATIC_ASSERT((is_computation_token< MssDescriptor >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<ExtentSizes>::type::value==0 || is_sequence_of< ExtentSizes, is_extent >::value), "Internal Error: wrong type");
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
        typedef typename boost::mpl::fold<
            boost::mpl::range_c< ushort_t, 0, boost::mpl::size< functors_seq_t >::value >,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                functor_id_pair< boost::mpl::_2, boost::mpl::at< functors_seq_t, boost::mpl::_2 >, RepeatFunctor > > >::
            type functors_list_t;

        typedef ExtentSizes extent_sizes_t;
        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;
    };

    template < typename T >
    struct is_mss_components : boost::mpl::false_ {};

    template < typename MssDescriptor, typename ExtentSizes, typename RepeatFunctor >
    struct is_mss_components< mss_components< MssDescriptor, ExtentSizes, RepeatFunctor > > : boost::mpl::true_ {};

} // namespace gridtools
