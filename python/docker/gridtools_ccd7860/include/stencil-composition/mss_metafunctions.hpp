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
#include "esf.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"

namespace gridtools {

template<
    typename functors_list,
    typename Coords
>
struct mss_intervals
{
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        functors_list,
        compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
    >::type functor_do_methods; // Vector of vectors - each element is a vector of pairs of actual axis-indices

    /**
     * compute the loop intervals
     */
    typedef typename compute_loop_intervals<
        functor_do_methods,
        typename Coords::axis_type
    >::type loop_intervals_t; // vector of pairs of indices - sorted and contiguous

    /**
     * compute the do method lookup maps
     *
     */
    typedef typename boost::mpl::transform<
        functor_do_methods,
        compute_functor_do_method_lookup_map<boost::mpl::_, loop_intervals_t>
    >::type functor_do_method_lookup_maps; // vector of maps, indexed by functors indices in Functor vector.
};

/**
 * @struct is_mss_parameter
 * metafunction that determines if a given type is a valid parameter for mss_descriptor
 */
template<typename T>
struct is_mss_parameter
{
    typedef typename boost::mpl::or_< is_sequence_of<T, is_cache >, is_esf_descriptor<T> >::type type;
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
