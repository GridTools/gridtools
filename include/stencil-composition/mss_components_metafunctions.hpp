 #pragma once
#include <boost/mpl/assert.hpp>
#include "mss_metafunctions.hpp"
#include "mss_components.hpp"
#include "../common/meta_array.hpp"

namespace gridtools {

//TODOCOSUNA unittest this
/**
 * @brief metafunction that takes an MSS with multiple ESFs and split it into multiple MSS with one ESF each
 * @tparam MssArray meta array of MSS
 */
template<typename MssArray>
struct split_esfs_into_independent_mss
{
    GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssArray, is_mss_descriptor>::value), "Internal Error: wrong type");

    template<typename MssDescriptor>
    struct mss_split_esfs
    {
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: wrong type");

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
        typename boost::mpl::reverse_fold<
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


/**
 * @brief metafunction that builds the array of mss components
 * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
 * @tparam MssDescriptorArray meta array of mss descriptors
 * @tparam range_sizes sequence of sequence of ranges
 */
template<
    enumtype::backend BackendId,
    typename MssDescriptorArray,
    typename RangeSizes
>
struct build_mss_components_array
{
    GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssDescriptorArray, is_mss_descriptor>::value), "Internal Error: wrong type");

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename MssDescriptorArray::elements>::value ==
                             boost::mpl::size<RangeSizes>::value), "Internal Error: wrong size");

    template<typename _RangeSizes_>
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
            _RangeSizes_,
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
        boost::mpl::identity<RangeSizes>,
        unroll_range_sizes<RangeSizes>
    >::type range_sizes_unrolled_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename mss_array_t::elements>::value ==
        boost::mpl::size<range_sizes_unrolled_t>::value
                                ), "Internal Error: wrong size");

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

/**
 * @brief metafunction that computes the mss functor do methods
 */
template<
    typename MssComponents,
    typename Coords
>
struct mss_functor_do_methods
{
    GRIDTOOLS_STATIC_ASSERT((is_mss_components<MssComponents>::value), "Internal Error: wrong type");

    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        typename MssComponents::functors_list_t,
        compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
    >::type type; // Vector of vectors - each element is a vector of pairs of actual axis-indices
};

/**
 * @brief metafunction that computes the loop intervals of an mss
 */
template<
    typename MssComponents,
    typename Coords
>
struct mss_loop_intervals
{
    GRIDTOOLS_STATIC_ASSERT((is_mss_components<MssComponents>::value), "Internal Error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");

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
    GRIDTOOLS_STATIC_ASSERT((is_mss_components<MssComponents>::value), "Internal Error: wrong type");
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
