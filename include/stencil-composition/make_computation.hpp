#pragma once

#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/ref.hpp>


#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include "stencil-composition/backend.hpp"
#include "stencil-composition/esf.hpp"
#include "stencil-composition/mss_metafunctions.hpp"
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "intermediate.hpp"
#include "../common/meta_array.hpp"
#include "caches/define_caches.hpp"

#ifndef NDEBUG

#ifndef __CUDACC__
#define POSITIONAL_WHEN_DEBUGGING true
#ifndef SUPPRESS_MESSAGES
#pragma message (">>\n>> In debug mode each computation is positional,\n>> so the loop indices can be queried from within\n>> the operator functions")
#endif
#else
#define POSITIONAL_WHEN_DEBUGGING false
#endif
#else
#define POSITIONAL_WHEN_DEBUGGING false
#endif

namespace gridtools {

namespace _impl {
    /**
     * @brief metafunction that extracts a meta array with all the mss descriptors found in the Sequence of types
     * @tparam Sequence sequence of types that contains some mss descriptors
     */
    template<typename Sequence>
    struct get_mss_array
    {
        GRIDTOOLS_STATIC_ASSERT(( boost::mpl::is_sequence<Sequence>::value ), "Internal Error: wrong type");

        typedef typename boost::mpl::fold<
            Sequence,
            boost::mpl::vector0<>,
            boost::mpl::eval_if<
                is_amss_descriptor<boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>,
                boost::mpl::_1
            >
        >::type mss_vector;

        typedef meta_array<mss_vector, boost::mpl::quote1<is_amss_descriptor> > type;
    };

    //check in a sequence of AMss that if there is reduction, it is placed at the end
    template<typename AMssSeq>
    struct check_mss_seq
    {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<AMssSeq, is_amss_descriptor>::value), "Error");
        typedef typename boost::mpl::find_if<AMssSeq, is_reduction_descriptor<boost::mpl::_> >::type check_iter_t;

        GRIDTOOLS_STATIC_ASSERT((boost::is_same<check_iter_t, typename boost::mpl::end<AMssSeq>::type >::value ||
                                 check_iter_t::pos::value == boost::mpl::size<AMssSeq>::value - 1),
                                "Error deducing the reduction. Check that if there is a reduction, this appears in the last mss");
        typedef notype type;
    };

    /**
     * helper struct to deduce the type of a reduction and extract the initial value of a reduction passed via API.
     * specialization returns a notype when argument passed is not a reduction
     */
    template<typename Mss>
    struct reduction_helper;

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct reduction_helper<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        typedef mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> mss_t;
        typedef notype reduction_type_t;
        static notype extract_initial_value(mss_t) { return 0;}
    };

    template <typename ExecutionEngine,
              typename BinOp,
              typename EsfDescrSequence>
    struct reduction_helper<reduction_descriptor<ExecutionEngine, BinOp, EsfDescrSequence> >
    {
        typedef reduction_descriptor<ExecutionEngine, BinOp, EsfDescrSequence> mss_t;
        typedef typename mss_t::reduction_type_t reduction_type_t;

        static typename mss_t::reduction_type_t extract_initial_value(mss_t& red) { return red.get();}
    };

} //namespace _impl

#define _MSS_DECL(z, n, nil)                                            \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(MssType, n ) BOOST_PP_CAT(mss, n )

#ifdef __CUDACC__

#define _MAKE_COMPUTATION(z, n, nil)                                                \
    template <                                                                      \
        typename Backend,                                                           \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                    \
        typename Domain,                                                            \
        typename Grid                                                               \
    >                                                                               \
    computation<typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t>* \
    make_computation(                                                               \
        BOOST_PP_REPEAT( BOOST_PP_INC(n), _MSS_DECL, _),                            \
        Domain& domain, const Grid& grid                                            \
    ) {                                                                             \
        typedef typename _impl::check_mss_seq<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>   \
        >::type nothing_t;                                                          \
                                                                                    \
        return new intermediate<                                                    \
            Backend,                                                                \
            typename _impl::get_mss_array<                                          \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                                \
            Domain, Grid,                                                           \
            typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t, \
            POSITIONAL_WHEN_DEBUGGING                                               \
        >(boost::ref(domain), grid,                                                 \
            _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::extract_initial_value(BOOST_PP_CAT(mss,n)) );         \
    }

#else

#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                           \
    >                                                                           \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid ,                                                      \
            typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t, \
            POSITIONAL_WHEN_DEBUGGING                                           \
        >                                                                       \
    > make_computation(                                                         \
        BOOST_PP_REPEAT( BOOST_PP_INC(n), _MSS_DECL, _),                        \
        Domain& domain, const Grid& grid                                        \
    ) {                                                                         \
        typedef typename _impl::check_mss_seq<                                   \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>   \
        >::type nothing_t;                                                      \
                                                                                \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Grid,                                                   \
                typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t, \
                POSITIONAL_WHEN_DEBUGGING                                       \
            >                                                                   \
        >(boost::ref(domain), grid,                                             \
            _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::extract_initial_value(BOOST_PP_CAT(mss,n)) );             \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION, _)
#undef _MAKE_COMPUTATION


    /////////////////////////////////////////////////////////////////////////////////////
    /// MAKE POSITIONAL COMPUTATIOS
    ////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                 \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                           \
    >                                                                           \
    computation<typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t>*  \
    make_positional_computation(                                                \
        BOOST_PP_REPEAT( BOOST_PP_INC(n), _MSS_DECL, _),                        \
        Domain& domain, const Grid& grid                                        \
    ) {                                                                         \
        typedef typename _impl::check_mss_seq<                                   \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>   \
        >::type nothing_t;                                                      \
                                                                                \
        return new intermediate<                                                \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid,                                                       \
            typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t,  \
            true                                                                \
        >(boost::ref(domain), grid,                                             \
             _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::extract_initial_value(BOOST_PP_CAT(mss,n)) );                      \
    }

#else

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                 \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                           \
    >                                                                           \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid ,                                                      \
            typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t, \
            true                                                                \
        >                                                                       \
    > make_positional_computation(                                              \
        BOOST_PP_REPEAT( BOOST_PP_INC(n), _MSS_DECL, _),                        \
        Domain& domain, const Grid& grid                                        \
    ) {                                                                         \
        typedef typename _impl::check_mss_seq<                                   \
           BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>   \
        >::type nothing_t;                                                      \
                                                                                \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Grid,                                                   \
                typename _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::reduction_type_t, \
                true                                                            \
            >                                                                   \
        >(boost::ref(domain), grid,                                             \
                _impl::reduction_helper< BOOST_PP_CAT(MssType,n) >::extract_initial_value(BOOST_PP_CAT(mss,n)) );                   \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_POSITIONAL_COMPUTATION, _)
#undef _MAKE_POSITIONAL_COMPUTATION

} //namespace gridtools
