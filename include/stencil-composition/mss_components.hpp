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
    template < typename T1, typename T2 >
    struct functor_id_pair {
        typedef T1 id;
        typedef T2 f_type;
    };

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     */
    template < typename MssDescriptor, typename ExtentSizes >
    struct mss_components {
        GRIDTOOLS_STATIC_ASSERT((is_amss_descriptor< MssDescriptor >::value), "Internal Error: wrong type");

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ExtentSizes, is_extent >::value), "Internal Error: wrong type");
        typedef MssDescriptor mss_descriptor_t;

        typedef typename mss_descriptor_execution_engine< MssDescriptor >::type execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        typedef typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type linear_esf_t;

        /** Compute a vector of vectors of temp indices of temporaries initialized by each functor*/
        typedef typename boost::mpl::fold< linear_esf_t,
            boost::mpl::vector<>,
            boost::mpl::push_back< boost::mpl::_1, esf_get_w_temps_per_functor< boost::mpl::_2 > > >::type
            written_temps_per_functor_t;

        /**
         * typename linear_esf is a list of all the esf nodes in the multi-stage descriptor.
         * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
         */
        typedef typename boost::mpl::transform< linear_esf_t, _impl::extract_functor >::type functors_seq_t;

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
                functor_id_pair< boost::mpl::_2, boost::mpl::at< functors_seq_t, boost::mpl::_2 > > > >::type
            functors_list_t;

        typedef ExtentSizes extent_sizes_t;
        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;
    };

    template < typename T >
    struct is_mss_components : boost::mpl::false_ {};

    template < typename MssDescriptor, typename ExtentSizes >
    struct is_mss_components< mss_components< MssDescriptor, ExtentSizes > > : boost::mpl::true_ {};

} // namespace gridtools
