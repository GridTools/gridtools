#include "conditionals/conditionals_macros_cxx03.hpp"
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif

namespace gridtools {

    namespace _impl {

        /**
         * @brief metafunction that extracts a meta array with all the mss descriptors found in the Sequence of types
         * @tparam Sequence sequence of types that contains some mss descriptors
         */
        template < typename Sequence >
        struct get_mss_array {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence< Sequence >::value), "Internal Error: wrong type");

            typedef typename boost::mpl::fold< Sequence,
                boost::mpl::vector0<>,
                boost::mpl::eval_if< is_mss_descriptor< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type mss_vector;

            typedef meta_array< mss_vector, boost::mpl::quote1< is_mss_descriptor > > type;
        };
    } // namespace _impl

#define _PAIR_(count, N, data) data##Type##N data##Value##N

#ifdef __CUDACC__
#define _POINTER_ computation *
#else
#define _POINTER_ boost::shared_ptr< computation >
#endif

#define _MAKE_COMPUTATION(z, n, nil)                                                                            \
    template < typename Backend,                                                                                \
        typename Domain,                                                                                        \
        typename Grid,                                                                                          \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                               \
    _POINTER_ make_computation(Domain &domain, const Grid &grid, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) { \
        return make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(                                     \
            domain, grid, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                                     \
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION, _)
#undef _MAKE_COMPUTATION

/////////////////////////////////////////////////////////////////////////////////////
/// MAKE POSITIONAL COMPUTATIOS
////////////////////////////////////////////////////////////////////////////////////

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                                                       \
    template < typename Backend,                                                                                      \
        typename Domain,                                                                                              \
        typename Grid,                                                                                                \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                                     \
    _POINTER_ make_positional_computation(                                                                            \
        Domain &domain, const Grid &grid, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) {                              \
        return make_computation_impl< true, Backend >(domain, grid, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue)); \
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_POSITIONAL_COMPUTATION, _)
#undef _MAKE_POSITIONAL_COMPUTATION
#undef _PAIR_
#undef _POINTER_
} // namespace gridtools
