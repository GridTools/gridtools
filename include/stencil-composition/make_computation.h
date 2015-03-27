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

#include "common/defs.h"
#include "stencil-composition/esf.h"
#include "stencil-composition/mss.h"
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "intermediate.h"
#include "../common/meta_array.h"

namespace gridtools {

namespace _impl {
    /**
     * @brief metafunction that extracts a meta array with all the mss descriptors found in the Sequence of types
     * @tparam Sequence sequence of types that contains some mss descriptors
     */
    template<typename Sequence>
    struct get_mss_array
    {
        BOOST_STATIC_ASSERT(( boost::mpl::is_sequence<Sequence>::value ));

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
        typename LayoutType,                                                    \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords \
    >                                                                           \
    computation* make_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            LayoutType,                                                         \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type, \
            Domain, Coords,false                                                      \
        >(boost::ref(domain), coords);                                          \
    }

#else

#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        typename LayoutType,                                                    \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords \
    >                                                                   \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            LayoutType,                                                         \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords ,false \
        >                                                                       \
    > make_computation(                                                         \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                LayoutType,                                                     \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type, \
                Domain, Coords, false                                                  \
            >                                                                   \
        >(boost::ref(domain), coords);                                          \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION, _)
#undef _MAKE_COMPUTATION


    /////////////////////////////////////////////////////////////////////////////////////
    /// MAKE STATEFUL COMPUTATIOS
    ////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

#define _MAKE_STATEFUL_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        typename LayoutType,                                                    \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords \
    >                                                                           \
    computation* make_stateful_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            LayoutType,                                                         \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type, \
            Domain, Coords,true                                                      \
        >(boost::ref(domain), coords);                                          \
    }

#else

#define _MAKE_STATEFUL_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        typename LayoutType,                                                    \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Coords \
    >                                                                   \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            LayoutType,                                                         \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Coords ,true \
        >                                                                       \
    > make_stateful_computation(                                                         \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Coords& coords                                    \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                LayoutType,                                                     \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type, \
                Domain, Coords, true                                                  \
            >                                                                   \
        >(boost::ref(domain), coords);                                          \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_STATEFUL_COMPUTATION, _)
#undef _MAKE_STATEFUL_COMPUTATION

} //namespace gridtools
