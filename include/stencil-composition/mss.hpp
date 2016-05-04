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


    namespace _impl {
        /*
          This metafunction traverses an array of esfs that may contain indendent_esfs.

          This is used by mss_descriptor_linear_esf_sequence and
          sequence_of_is_independent_esf.  The first one creates a
          vector of esfs the second one a vector of boolean types that
          are true if an esf is in an independent section, and false
          otherwise.

          To reuse the code, the values to push are passed as template
          arguments. By passing a placeholder as value to push, then
          the corresponding folds will "substitute" that, thus
          allowing to process sequence elements. Otherwise boolean
          types are passed and those will be pushed into the result
          vector. (This is a rework of Paolo's code that was failing
          for nested independents.
         */
        template <typename EsfsVector, typename PushRegular, typename PushIndependent=PushRegular>
        struct linearize_esf_array {

            template <typename Vector, typename Element>
            struct push_into {
                typedef typename boost::mpl::push_back<Vector, Element>::type type;
            };

            template <typename Vector, typename Independents>
            struct push_into<Vector, independent_esf<Independents> > {
                typedef typename boost::mpl::fold<
                    Independents,
                    Vector,
                    push_into<boost::mpl::_1, PushIndependent>
                    >::type type;
            };

            typedef typename boost::mpl::fold<
                EsfsVector,
                boost::mpl::vector0<>,
                push_into<boost::mpl::_1, PushRegular>
                >::type type;
        };
    } // namespace _impl

    /**
       @brief constructs an mpl vector of esf, linearizig the mss tree.

       Looping over all the esfs at compile time.
       if found independent esfs, they are also included in the linearized vector with a nested fold.

       NOTE: the nested make_independent calls get also linearized
     */
    template < typename T >
    struct mss_descriptor_linear_esf_sequence;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_linear_esf_sequence<
        mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence >
        >
    {
        typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::mpl::_2 >::type type;
    };


    /**
       @brief constructs an mpl vector of booleans, linearizing the mss tree and attachnig a true or false flag
       depending wether the esf is independent or not

       the code is very similar as in the metafunction above
     */
    template < typename T >
    struct sequence_of_is_independent_esf;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct sequence_of_is_independent_esf<
        mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence >
        >
    {
        typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::false_type, boost::true_type >::type type;
    };

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
