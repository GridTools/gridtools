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
/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/mpl/equal.hpp>
#include "functor_do_methods.hpp"
#include "loopintervals.hpp"
#include "functor_do_method_lookup_maps.hpp"
#include "caches/cache.hpp"
#include "caches/cache_metafunctions.hpp"
#include "stencil_composition/esf.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil_composition/caches/cache_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "mss.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "computation_grammar.hpp"

namespace gridtools {

    /**
     * @struct is_mss_parameter
     * metafunction that determines if a given type is a valid parameter for mss_descriptor
     */
    template < typename T >
    struct is_mss_parameter {
        typedef typename boost::mpl::or_< is_sequence_of< T, is_cache >, is_esf_descriptor< T > >::type type;
    };

    /**
     * @struct extract_mss_caches
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all caches
     */
    template < typename MssParameterSequence >
    struct extract_mss_caches {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssParameterSequence, is_mss_parameter >::value),
            "wrong set of mss parameters passed to make_mss construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_stage(...) or make_independent(...)");
        template < typename T >
        struct is_sequence_of_caches {
            typedef typename is_sequence_of< T, is_cache >::type type;
        };

#ifdef __DISABLE_CACHING__
        typedef boost::mpl::vector0<> type;
#else
        typedef typename boost::mpl::copy_if< MssParameterSequence, boost::mpl::quote1< is_sequence_of_caches > >::type
            sequence_of_caches;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< sequence_of_caches >::value == 1 || boost::mpl::size< sequence_of_caches >::value == 0),
            "Wrong number of sequence of caches. Probably caches are defined in multiple dinstinct instances of "
            "define_caches\n"
            "Only one instance of define_caches is allowed.");

        typedef typename boost::mpl::eval_if< boost::mpl::empty< sequence_of_caches >,
            boost::mpl::identity< boost::mpl::vector0<> >,
            boost::mpl::front< sequence_of_caches > >::type type;
#endif
    };

    /**
     * @struct extract_mss_esfs
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all esf descriptors
     */
    template < typename MssParameterSequence >
    struct extract_mss_esfs {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssParameterSequence, is_mss_parameter >::value),
            "wrong set of mss parameters passed to make_mss construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_stage(...) or make_independent(...)");
        typedef
            typename boost::mpl::copy_if< MssParameterSequence, boost::mpl::quote1< is_esf_descriptor > >::type type;
    };

    template < typename Mss1, typename Mss2 >
    struct mss_equal {
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< Mss1 >::value), "Error");
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< Mss2 >::value), "Error");

        typedef static_bool< ((boost::is_same< typename mss_descriptor_execution_engine< Mss1 >::type,
                                  typename mss_descriptor_execution_engine< Mss2 >::type >::value) &&
                              (boost::mpl::equal< typename mss_descriptor_esf_sequence< Mss1 >::type,
                                  typename mss_descriptor_esf_sequence< Mss2 >::type,
                                  esf_equal< boost::mpl::_1, boost::mpl::_2 > >::value) &&
                              (Mss1::is_reduction_t::value == Mss2::is_reduction_t::value) &&
                              (boost::mpl::equal< typename mss_descriptor_cache_sequence< Mss1 >::type,
                                  typename mss_descriptor_cache_sequence< Mss2 >::type >::value)) > type;
    };

} // namespace gridtools
