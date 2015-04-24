#pragma once
#include "mss.h"

namespace gridtools {

template<
    typename MssDescriptor,
    typename RangeSizes
>
struct mss_components
{
    BOOST_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value));

//    typedef ArrayEsfDescr esf_array_t; // may contain independent constructs
    typedef typename mss_descriptor_execution_engine<MssDescriptor>::type execution_engine_t;

    /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
        esf structs. Independent functors are listed one after the other.*/
    typedef typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type linear_esf_t;

    /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
    typedef typename boost::mpl::fold<linear_esf_t,
            boost::mpl::vector<>,
            boost::mpl::push_back<boost::mpl::_1, get_temps_per_functor<boost::mpl::_2> >
    >::type written_temps_per_functor_t;

    /**
     * typename linear_esf is a list of all the esf nodes in the multi-stage descriptor.
     * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
     */
    typedef typename boost::mpl::transform<
        linear_esf_t,
        _impl::extract_functor
    >::type functors_list_t;

    typedef RangeSizes range_sizes_t;
};

template<typename T> struct is_mss_components : boost::mpl::false_{};

template<
    typename MssDescriptor,
    typename RangeSizes
>
struct is_mss_components<mss_components<MssDescriptor, RangeSizes> > : boost::mpl::true_{};

} //namespace gridtools
