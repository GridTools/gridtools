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
#include "stencil-composition/esf.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "mss.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "computation_grammar.hpp"

namespace gridtools {

//     /**
//        @brief constructs an mpl vector of esf, linearizig the mss tree.

//        Looping over all the esfs at compile time.
//        if found independent esfs, they are also included in the linearized vector with a nested fold.

//        NOTE: the nested make_independent calls get also linearized
//      */
//     template < typename AMssDescriptor >
//     struct mss_descriptor_linear_esf_sequence {
//         GRIDTOOLS_STATIC_ASSERT((is_computation_token< AMssDescriptor >::value), "Error");

//         template < typename State, typename SubArray >
//         struct keep_scanning : keep_scanning_lambda< State, SubArray, boost::mpl::_2 > {};

//         template < typename Array >
//         struct linearize_esf_array : linearize_esf_array_lambda< Array, boost::mpl::_2, keep_scanning > {};

//         typedef typename linearize_esf_array< typename AMssDescriptor::esf_sequence_t >::type type;
//     };

//     /**
//        @brief constructs an mpl vector of booleans, linearizing the mss tree and attachnig a true or false flag
//        depending wether the esf is independent or not

//        the code is very similar as in the metafunction above
//      */
//     template < typename T >
//     struct sequence_of_is_independent_esf;

//     template < typename AMssDescriptor >
//     struct sequence_of_is_independent_esf {
//         GRIDTOOLS_STATIC_ASSERT((is_computation_token< AMssDescriptor >::value), "Error");

//         template < typename State, typename SubArray >
//         struct keep_scanning : keep_scanning_lambda< State, SubArray, boost::mpl::true_ > {};

//         template < typename Array >
//         struct linearize_esf_array : linearize_esf_array_lambda< Array, boost::mpl::false_, keep_scanning > {};

//         typedef typename linearize_esf_array< typename AMssDescriptor::esf_sequence_t >::type type;
//     };

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
            " * esf descriptors from make_esf(...) or make_independent(...)");
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
            " * esf descriptors from make_esf(...) or make_independent(...)");
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
