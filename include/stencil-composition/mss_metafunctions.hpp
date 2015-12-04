/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once
#include "functor_do_methods.hpp"
#include "loopintervals.hpp"
#include "functor_do_method_lookup_maps.hpp"
#include "caches/cache.hpp"
#include "caches/cache_metafunctions.hpp"
#include "stencil-composition/esf.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"

namespace gridtools {

/**
 * @struct is_mss_parameter
 * metafunction that determines if a given type is a valid parameter for mss_descriptor
 */
template<typename T>
struct printi{BOOST_MPL_ASSERT_MSG((false), YYYYYYYYYYYY, (T));};
template <typename T>
struct testt
{
    printi<T> oi;
};

template<typename T>
struct is_mss_parameter
{

    typedef typename boost::mpl::or_< is_sequence_of<T, is_cache >, is_esf_descriptor<T> >::type type;
    typedef typename boost::mpl::eval_if<
        type,
        boost::mpl::identity<boost::mpl::true_>,
        testt<T>
    >::type OO;


};

/**
 * @struct extract_mss_caches
 * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all caches
 */
template<typename MssParameterSequence>
struct extract_mss_caches
{
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssParameterSequence, is_mss_parameter >::value),
            "wrong set of mss parameters passed to make_mss construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_esf(...) or make_independent(...)");
    template<typename T>
    struct is_sequence_of_caches{
        typedef typename is_sequence_of<T, is_cache>::type type;
    };

#ifdef __DISABLE_CACHING__
    typedef boost::mpl::vector0<> type;
#else
    typedef typename boost::mpl::copy_if<MssParameterSequence, boost::mpl::quote1<is_sequence_of_caches> >::type sequence_of_caches;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<sequence_of_caches>::value==1 || boost::mpl::size<sequence_of_caches>::value==0),
        "Wrong number of sequence of caches. Probably caches are defined in multiple dinstinct instances of define_caches\n"
        "Only one instance of define_caches is allowed." );

    typedef typename boost::mpl::eval_if<
        boost::mpl::empty<sequence_of_caches>,
        boost::mpl::identity<boost::mpl::vector0<> >,
        boost::mpl::front<sequence_of_caches>
    >::type type;
#endif
};

/**
 * @struct extract_mss_esfs
 * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all esf descriptors
 */
template<typename MssParameterSequence>
struct extract_mss_esfs
{
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssParameterSequence, is_mss_parameter>::value),
            "wrong set of mss parameters passed to make_mss construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_esf(...) or make_independent(...)");
    typedef typename boost::mpl::copy_if<MssParameterSequence, boost::mpl::quote1<is_esf_descriptor> >::type type;
};

} //namespace gridtools
