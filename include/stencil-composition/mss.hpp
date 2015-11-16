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


    /**
       @brief cnostructs an mpl vector of esf, linearizig the mss tree.

       Looping over all the esfs at compile time.
       if found independent esfs, they are also included in the linearized vector with a nested fold.
       NOTE: nested independent sets are not supported (why?), should trigger an error
     */
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





    template<typename T>
    struct is_independent_esf_sequence;

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct is_independent_esf_sequence<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        template <typename State, typename SubArray>
        struct keep_scanning
          : boost::mpl::fold<
                typename SubArray::esf_list,
                State,
            boost::mpl::insert<boost::mpl::_1, boost::mpl::pair<extract_esf_function<boost::mpl::_2>, boost::mpl::true_> >
            >
        {};

        template <typename Array>
        struct linearize_esf_array : boost::mpl::fold<
              Array,
              boost::mpl::map<>,
              boost::mpl::if_<
                  is_independent<boost::mpl::_2>,
                  keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                  boost::mpl::insert<boost::mpl::_1, boost::mpl::pair<extract_esf_function<boost::mpl::_2>, boost::mpl::false_> >
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

} // namespace gridtools
