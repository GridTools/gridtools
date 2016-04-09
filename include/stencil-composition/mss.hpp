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
    namespace _impl {

        struct extract_functor {
            template < typename T >
            struct apply {
                typedef typename T::esf_function type;
            };
        };

        /**@brief Macro defining a sfinae metafunction

           defines a metafunction has_extent_type, which returns true if its template argument
           defines a type called extent_type. It also defines a get_extent_type metafunction, which
           can be used to return the extent_type only when it is present, without giving compilation
           errors in case it is not defined.
         */
        HAS_TYPE_SFINAE(extent_type, has_extent_type, get_extent_type)
    }

    template < typename Mss1, typename Mss2, typename Tag >
    struct condition;

    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence = boost::mpl::vector0<> >
    struct mss_descriptor {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< EsfDescrSequence, is_esf_descriptor >::value), "Internal Error: invalid type");

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Internal Error: invalid type");
        typedef EsfDescrSequence esf_sequence_t;
        typedef CacheSequence cache_sequence_t;
        typedef static_bool< false > is_reduction_t;
    };

    template < typename mss >
    struct is_mss_descriptor : boost::mpl::false_ {};

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct is_mss_descriptor< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > : boost::mpl::true_ {
    };

    template < typename Mss1, typename Mss2, typename C >
    struct is_mss_descriptor< condition< Mss1, Mss2, C > >
        : boost::mpl::and_< is_mss_descriptor< Mss1 >, is_mss_descriptor< Mss2 > >::type {};

    template < typename Mss >
    struct mss_descriptor_esf_sequence {};

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_esf_sequence< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef EsfDescrSequence type;
    };

    template < typename Mss >
    struct mss_descriptor_cache_sequence {};

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_cache_sequence< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef CacheSequence type;
    };

    template < typename Mss >
    struct mss_descriptor_is_reduction;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_is_reduction< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef static_bool< false > type;
    };

    /**
       @brief pushes an element in a vector based on the fact that an ESF is independent or not

       Helper metafunction, used by other metafunctions
     */
    template < typename State, typename SubArray, typename VectorComponent >
    struct keep_scanning_lambda
        : boost::mpl::fold< typename SubArray::esf_list,
              State,
              boost::mpl::if_< is_independent< boost::mpl::_2 >,
                                keep_scanning_lambda< boost::mpl::_1, boost::mpl::_2, VectorComponent >,
                                boost::mpl::push_back< boost::mpl::_1, VectorComponent > > > {};

    /**
       @brief linearizes the ESF tree and returns a vector

       Helper metafunction, used by other metafunctions
     */
    template < typename Array, typename Argument, template < typename, typename > class KeepScanning >
    struct linearize_esf_array_lambda : boost::mpl::fold< Array,
                                            boost::mpl::vector0<>,
                                            boost::mpl::if_< is_independent< boost::mpl::_2 >,
                                                              KeepScanning< boost::mpl::_1, boost::mpl::_2 >,
                                                              boost::mpl::push_back< boost::mpl::_1, Argument > > > {};

    template < typename Mss >
    struct mss_descriptor_execution_engine {};

    template < typename Mss1, typename Mss2, typename Cond >
    struct mss_descriptor_execution_engine< condition< Mss1, Mss2, Cond > > {
        typedef typename mss_descriptor_execution_engine< Mss1 >::type type;
    };

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_execution_engine< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef ExecutionEngine type;
    };

} // namespace gridtools
