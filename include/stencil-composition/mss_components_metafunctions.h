 #pragma once
#include <boost/mpl/assert.hpp>
#include "mss_metafunctions.h"
#include "mss_components.h"
#include "../common/meta_array.h"

namespace gridtools {

//template<
//    typename MssDescriptor,
//    typename range_sizes
//>
//struct create_mss_components
//{
//    BOOST_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value));
//
//    template<typename Esf>
//    struct extract_range
//    {
//        BOOST_STATIC_ASSERT((boost::mpl::has_key<range_sizes_map, Esf>::value));
//        typedef typename boost::mpl::at<range_sizes_map, Esf>::type type;
//    };
//
//    typedef typename boost::mpl::fold<
//        typename mss_descriptor_esf_sequence<MssDescriptor>::type,
//        boost::mpl::vector0<>,
//        boost::mpl::push_back<boost::mpl::_1, extract_range<boost::mpl::_2> >
//    >::type range_sizes_t;
//
//    typedef mss_components<MssDescriptor, range_sizes_t> type;
//
//};

template<typename MssArray>
struct split_esfs_into_independent_mss
{
    //TODOCOSUNA
    BOOST_MPL_ASSERT_MSG((is_meta_array_of<MssArray, is_mss_descriptor>::value), JJJJJJJJJJJJJJJJJJJJJJJJj, (MssArray));
//    BOOST_STATIC_ASSERT((is_meta_array_of<MssArray, is_mss_descriptor>::value));

    template<typename MssDescriptor>
    struct mss_split_esfs
    {
        BOOST_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value));

        typedef typename mss_descriptor_execution_engine<MssDescriptor>::type execution_engine_t;

        typedef typename boost::mpl::fold<
            typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                mss_descriptor<
                    execution_engine_t,
                    boost::mpl::vector1<boost::mpl::_2>
                >
            >
        >::type type;
    };

    typedef meta_array<
        typename boost::mpl::fold<
            typename MssArray::elements,
            boost::mpl::vector0<>,
            boost::mpl::copy<
                boost::mpl::_1,
                boost::mpl::back_inserter<mss_split_esfs<boost::mpl::_2> >
            >
        >::type,
        boost::mpl::quote1<is_mss_descriptor>
    >type;
};


template<
    enumtype::backend BackendId,
    typename MssDescriptorArray,
    typename range_sizes
>
struct build_mss_components_array
{
    BOOST_STATIC_ASSERT((is_meta_array_of<MssDescriptorArray, is_mss_descriptor>::value));
    BOOST_STATIC_ASSERT((boost::mpl::size<typename MssDescriptorArray::elements>::value ==
            boost::mpl::size<range_sizes>::value));

    template<typename RangeSizes>
    struct unroll_range_sizes
    {
        template<typename State, typename Sequence>
        struct insert_unfold
        {
            typedef typename boost::mpl::fold<
                Sequence,
                State,
                boost::mpl::push_back<
                    boost::mpl::_1,
                    boost::mpl::vector1<boost::mpl::_2>
                >
            >::type type;
        };
        typedef typename boost::mpl::fold<
            RangeSizes,
            boost::mpl::vector0<>,
            insert_unfold<boost::mpl::_1, boost::mpl::_2>
        >::type type;
    };

    typedef typename boost::mpl::eval_if<
        typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
        boost::mpl::identity<MssDescriptorArray>,
        split_esfs_into_independent_mss<MssDescriptorArray>
    >::type mss_array_t;

    typedef typename boost::mpl::eval_if<
        typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
        boost::mpl::identity<range_sizes>,
        unroll_range_sizes<range_sizes>
    >::type range_sizes_unrolled_t;

    BOOST_STATIC_ASSERT((boost::mpl::size<typename mss_array_t::elements>::value ==
        boost::mpl::size<range_sizes_unrolled_t>::value
    ));

    typedef meta_array<
        typename boost::mpl::fold<
            boost::mpl::range_c<int,0, boost::mpl::size<range_sizes_unrolled_t>::value>,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                mss_components<
                    boost::mpl::at<
                        typename mss_array_t::elements,
                        boost::mpl::_2
                    >,
                    boost::mpl::at<
                        range_sizes_unrolled_t,
                        boost::mpl::_2
                    >
                >
            >
        >::type,
        boost::mpl::quote1<is_mss_components>
    > type;
};

template<
    typename MssComponents,
    typename Coords
>
struct mss_functor_do_methods
{
    BOOST_STATIC_ASSERT((is_mss_components<MssComponents>::value));

    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        typename MssComponents::functors_list_t,
        compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
    >::type type; // Vector of vectors - each element is a vector of pairs of actual axis-indices
};

template<
    typename MssComponents,
    typename Coords
>
struct mss_loop_intervals
{
    BOOST_STATIC_ASSERT((is_mss_components<MssComponents>::value));
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename mss_functor_do_methods<MssComponents, Coords>::type functor_do_methods;

    /**
     * compute the loop intervals
     */
    typedef typename compute_loop_intervals<
        functor_do_methods,
        typename Coords::axis_type
    >::type type; // vector of pairs of indices - sorted and contiguous
};

template<
    typename MssComponents,
    typename Coords
>
struct mss_functor_do_method_lookup_maps
{
    BOOST_STATIC_ASSERT((is_mss_components<MssComponents>::value));
    typedef typename mss_functor_do_methods<MssComponents, Coords>::type functor_do_methods;

    typedef typename mss_loop_intervals<MssComponents, Coords>::type loop_intervals;
    /**
     * compute the do method lookup maps
     *
     */
    typedef typename boost::mpl::transform<
        functor_do_methods,
        compute_functor_do_method_lookup_map<boost::mpl::_, loop_intervals>
    >::type type; // vector of maps, indexed by functors indices in Functor vector.
};

} // namespace gridtools
