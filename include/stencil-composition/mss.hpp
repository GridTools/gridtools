#pragma once
#include <boost/mpl/transform.hpp>
#include <boost/mpl/map/map0.hpp>
#include <boost/mpl/assert.hpp>
#include "functor_do_methods.hpp"
#include "esf.hpp"
#include "../common/meta_array.hpp"
#include "caches/cache_metafunctions.hpp"
#include "independent_esf.hpp"
#include "stencil-composition/sfinae.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    namespace _impl
    {

        struct extract_functor {
            template <typename T>
            struct apply {
                typedef typename T::esf_function type;
            };
        };

        /**@brief Macro defining a sfinae metafunction

           defines a metafunction has_range_type, which returns true if its template argument
           defines a type called range_type. It also defines a get_range_type metafunction, which
           can be used to return the range_type only when it is present, without giving compilation
           errors in case it is not defined.
         */
        HAS_TYPE_SFINAE(range_type, has_range_type, get_range_type)

        /**@brief wrap type to simplify specialization based on mpl::vectors */
        template <typename MplArray>
        struct wrap_type {
            typedef MplArray type;
        };

        /**
         * @brief compile-time boolean operator returning true if the template argument is a wrap_type
         * */
        template <typename T>
        struct is_wrap_type : boost::false_type {};

        template <typename T>
        struct is_wrap_type<wrap_type<T> > : boost::true_type{};

    }

    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence = boost::mpl::vector0<> >
    struct mss_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfDescrSequence, is_esf_descriptor>::value), "Internal Error: invalid type");

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<CacheSequence, is_cache>::value),
                "Internal Error: invalid type");
        typedef EsfDescrSequence esf_sequence_t;
        typedef CacheSequence cache_sequence_t;
    };

    template<typename mss>
    struct is_mss_descriptor : boost::mpl::false_{};

    template <typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence>
    struct is_mss_descriptor<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> > : boost::mpl::true_{};

    template<typename Mss>
    struct mss_descriptor_esf_sequence {};

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_esf_sequence<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        typedef EsfDescrSequence type;
    };

    template<typename T>
    struct mss_descriptor_linear_esf_sequence;

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_linear_esf_sequence<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        template <typename State, typename SubArray>
        struct keep_scanning
          : boost::mpl::fold<
                typename SubArray::esf_list,
                State,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
            >
        {};

        template <typename Array>
        struct linearize_esf_array : boost::mpl::fold<
              Array,
              boost::mpl::vector<>,
              boost::mpl::if_<
                  is_independent<boost::mpl::_2>,
                  keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                  boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
              >
        >{};

        typedef typename linearize_esf_array<EsfDescrSequence>::type type;
    };

    template<typename Mss>
    struct mss_descriptor_execution_engine {};

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_execution_engine<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        typedef ExecutionEngine type;
    };

    template<typename MssDescriptor>
    struct mss_compute_range_sizes
    {
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: invalid type");

        /**
         * \brief Here the ranges are calculated recursively, in order for each functor's domain to embed all the domains of the functors he depends on.
         */
        typedef typename boost::mpl::fold<
            typename mss_descriptor_esf_sequence<MssDescriptor>::type,
            boost::mpl::vector0<>,
            _impl::traverse_ranges<boost::mpl::_1,boost::mpl::_2>
        >::type ranges_list;

        /*
         *  Compute prefix sum to compute bounding boxes for calling a given functor
         */
        typedef typename _impl::prefix_on_ranges<ranges_list>::type structured_range_sizes;

        /**
         * linearize the data flow graph
         *
         */
        typedef typename _impl::linearize_range_sizes<structured_range_sizes>::type type;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size<typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type>::value ==
             boost::mpl::size<type>::value), "Internal Error: wrong size");
    };

} // namespace gridtools
