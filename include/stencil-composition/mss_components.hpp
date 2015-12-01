#pragma once
#include "mss.hpp"
#include "esf_metafunctions.hpp"

namespace gridtools {

/**
 * @brief the mss components contains meta data associated to a mss descriptor.
 * All derived metadata is computed in this class
 * @tparam MssDescriptor the mss descriptor
 * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
 */
template<
    typename MssDescriptor,
    typename ExtentSizes
>
struct mss_components
{
    GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ExtentSizes, is_extent>::value), "Internal Error: wrong type");
    typedef MssDescriptor mss_descriptor_t;

    typedef typename mss_descriptor_execution_engine<MssDescriptor>::type execution_engine_t;

    /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
        esf structs. Independent functors are listed one after the other.*/
    typedef typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type linear_esf_t;

    /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
    typedef typename boost::mpl::fold<linear_esf_t,
            boost::mpl::vector<>,
            boost::mpl::push_back<boost::mpl::_1, esf_get_w_temps_per_functor<boost::mpl::_2> >
    >::type written_temps_per_functor_t;

    /**
     * typename linear_esf is a list of all the esf nodes in the multi-stage descriptor.
     * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
     */
    typedef typename boost::mpl::transform<
        linear_esf_t,
        _impl::extract_functor
    >::type functors_seq_t;

    /*
      @brief attaching an integer index to each functor

      This ensure that the types in the functors_list_t are unique
    */
    typedef typename boost::mpl::fold<
        boost::mpl::range_c<ushort_t, 0, boost::mpl::size<functors_seq_t>::value >
        , boost::mpl::vector0<>
        , boost::mpl::push_back<boost::mpl::_1, boost::mpl::pair<boost::mpl::_2, boost::mpl::at<functors_seq_t, boost::mpl::_2> > >
        >::type functors_list_t;

    typedef ExtentSizes extent_sizes_t;
    typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;
};

template<typename T> struct is_mss_components : boost::mpl::false_{};

template<
    typename MssDescriptor,
    typename ExtentSizes
>
struct is_mss_components<mss_components<MssDescriptor, ExtentSizes> > : boost::mpl::true_{};

} //namespace gridtools
