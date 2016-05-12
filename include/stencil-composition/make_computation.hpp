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
                is_mss_descriptor<boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>,
                boost::mpl::_1
            >
        >::type mss_vector;

        typedef meta_array<mss_vector, boost::mpl::quote1<is_mss_descriptor> > type;
    };
} //namespace _impl

#ifdef __CUDACC__

#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords                                                         \
    >                                                                           \
    computation* make_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords,POSITIONAL_WHEN_DEBUGGING                            \
        >(boost::ref(domain), coords);                                          \
    }

#else

#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords                                                         \
    >                                                                           \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords ,POSITIONAL_WHEN_DEBUGGING                           \
        >                                                                       \
    > make_computation(                                                         \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Coords, POSITIONAL_WHEN_DEBUGGING                       \
            >                                                                   \
        >(boost::ref(domain), coords);                                          \
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
        typename Coords                                                         \
    >                                                                           \
    computation* make_positional_computation(                                   \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords,true                                                 \
        >(boost::ref(domain), coords);                                          \
    }

#else

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                 \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords                                                         \
    >                                                                            \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords ,true                                                \
        >                                                                       \
    > make_positional_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Coords, true                                            \
            >                                                                   \
        >(boost::ref(domain), coords);                                          \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_POSITIONAL_COMPUTATION, _)
#undef _MAKE_POSITIONAL_COMPUTATION

} //namespace gridtools
