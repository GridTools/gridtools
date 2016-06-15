#pragma once
#include <boost/mpl/assert.hpp>
#include "mss_metafunctions.hpp"
#include "mss_components.hpp"
#include "reductions/reduction_descriptor.hpp"
#include "../common/meta_array.hpp"

namespace gridtools {

    template < typename T >
    struct mss_components_is_reduction;

    template < typename MssDescriptor, typename ExtentSizes >
    struct mss_components_is_reduction< mss_components< MssDescriptor, ExtentSizes > > : MssDescriptor::is_reduction_t {
    };

    // TODOCOSUNA unittest this
    /**
     * @brief metafunction that takes an MSS with multiple ESFs and split it into multiple MSS with one ESF each
     * Only to be used for CPU. GPU always fuses ESFs and there is no clear way to split the caches.
     * @tparam MssArray meta array of MSS
     */
    template < typename MssArray >
    struct split_mss_into_independent_esfs {
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssArray, is_amss_descriptor >::value), "Internal Error: wrong type");

        template < typename MssDescriptor >
        struct mss_split_esfs {
            GRIDTOOLS_STATIC_ASSERT((is_amss_descriptor< MssDescriptor >::value), "Internal Error: wrong type");

            typedef typename mss_descriptor_execution_engine< MssDescriptor >::type execution_engine_t;

            template < typename Esf_ >
            struct compose_mss_ {
                typedef mss_descriptor< execution_engine_t, boost::mpl::vector1< Esf_ > > type;
            };

            struct mss_split_multiple_esf {
                typedef typename boost::mpl::fold< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back< boost::mpl::_1, compose_mss_< boost::mpl::_2 > > >::type type;
            };

            typedef typename boost::mpl::if_c<
                // if the number of esf contained in the mss is 1, there is no need to split
                (boost::mpl::size< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type >::value == 1),
                boost::mpl::vector1< MssDescriptor >,
                typename mss_split_multiple_esf::type >::type type;
        };

        typedef meta_array<
            typename boost::mpl::reverse_fold< typename MssArray::elements,
                boost::mpl::vector0<>,
                boost::mpl::copy< boost::mpl::_1, boost::mpl::back_inserter< mss_split_esfs< boost::mpl::_2 > > > >::
                type,
            boost::mpl::quote1< is_amss_descriptor > > type;
    };

    /**
     * @brief metafunction that builds the array of mss components
     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptorArray meta array of mss descriptors
     * @tparam extent_sizes sequence of sequence of extents
     */
    template < enumtype::platform BackendId, typename MssDescriptorArray, typename ExtentSizes >
    struct build_mss_components_array {
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_amss_descriptor >::value), "Internal Error: wrong type");

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename MssDescriptorArray::elements >::value ==
                                    boost::mpl::size< ExtentSizes >::value),
            "Internal Error: wrong size");

        template < typename _ExtentSizes_ >
        struct unroll_extent_sizes {
            template < typename State, typename Sequence >
            struct insert_unfold {
                typedef typename boost::mpl::fold< Sequence,
                    State,
                    boost::mpl::push_back< boost::mpl::_1, boost::mpl::vector1< boost::mpl::_2 > > >::type type;
            };

            typedef typename boost::mpl::fold< _ExtentSizes_,
                boost::mpl::vector0<>,
                insert_unfold< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            boost::mpl::identity< MssDescriptorArray >,
            split_mss_into_independent_esfs< MssDescriptorArray > >::type mss_array_t;

        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            boost::mpl::identity< ExtentSizes >,
            unroll_extent_sizes< ExtentSizes > >::type extent_sizes_unrolled_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename mss_array_t::elements >::value ==
                                    boost::mpl::size< extent_sizes_unrolled_t >::value),
            "wrong size of the arg_type vector defined inside at least one of the user functions");

        typedef meta_array<
            typename boost::mpl::fold<
                boost::mpl::range_c< int, 0, boost::mpl::size< extent_sizes_unrolled_t >::value >,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                    mss_components< boost::mpl::at< typename mss_array_t::elements, boost::mpl::_2 >,
                                           boost::mpl::at< extent_sizes_unrolled_t, boost::mpl::_2 > > > >::type,
            boost::mpl::quote1< is_mss_components > > type;
    }; // struct build_mss_components_array

    /**
     * @brief metafunction that builds a pair of arrays of mss components, to be handled at runtime
     via conditional switches

     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptorArray1 meta array of mss descriptors
     * @tparam MssDescriptorArray2 meta array of mss descriptors
     * @tparam extent_sizes sequence of sequence of extents
     */
    template < enumtype::platform BackendId,
        typename MssDescriptorArray1,
        typename MssDescriptorArray2,
        typename Predicate,
        typename Condition,
        typename ExtentSizes1,
        typename ExtentSizes2 >
    struct build_mss_components_array< BackendId,
        meta_array< condition< MssDescriptorArray1, MssDescriptorArray2, Condition >, Predicate >,
        condition< ExtentSizes1, ExtentSizes2, Condition > > {
        // typedef typename pair<
        //     typename build_mss_components_array<BackendId, MssDescriptorArray1, ExtentSizes>::type
        //     , typename build_mss_components_array<BackendId, MssDescriptorArray1, ExtentSizes>::type >
        // ::type type;
        typedef condition< typename build_mss_components_array< BackendId,
                               meta_array< MssDescriptorArray1, Predicate >,
                               ExtentSizes1 >::type,
            typename build_mss_components_array< BackendId,
                               meta_array< MssDescriptorArray2, Predicate >,
                               ExtentSizes2 >::type,
            Condition > type;
    }; // build_mss_components_array

    /**
     * @brief metafunction that computes the mss functor do methods
     */
    template < typename MssComponents, typename Grid >
    struct mss_functor_do_methods {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error: wrong type");

        /**
         *  compute the functor do methods - This is the most computationally intensive part
         */
        template < typename Functor >
        struct inserter_ {
            typedef typename compute_functor_do_methods< Functor, typename Grid::axis_type >::type type;
        };

        typedef typename boost::mpl::transform< typename MssComponents::functors_seq_t,
            inserter_< boost::mpl::_ > >::type
            type; // Vector of vectors - each element is a vector of pairs of actual axis-indices
    };

    /**
     * @brief metafunction that computes the loop intervals of an mss
     */
    template < typename MssComponents, typename Grid >
    struct mss_loop_intervals {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");

        /**
         *  compute the functor do methods - This is the most computationally intensive part
         */
        typedef typename mss_functor_do_methods< MssComponents, Grid >::type functor_do_methods;

        /**
         * compute the loop intervals
         */
        typedef typename compute_loop_intervals< functor_do_methods,
            typename Grid::axis_type >::type type; // vector of pairs of indices - sorted and contiguous
    };

    template < typename MssComponents, typename Grid >
    struct mss_functor_do_method_lookup_maps {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error: wrong type");
        typedef typename mss_functor_do_methods< MssComponents, Grid >::type functor_do_methods;

        typedef typename mss_loop_intervals< MssComponents, Grid >::type loop_intervals;
        /**
         * compute the do method lookup maps
         *
         */
        typedef typename boost::mpl::transform< functor_do_methods,
            compute_functor_do_method_lookup_map< boost::mpl::_, loop_intervals > >::type
            type; // vector of maps, indexed by functors indices in Functor vector.
    };

} // namespace gridtools
