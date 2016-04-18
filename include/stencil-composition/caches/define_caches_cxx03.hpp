#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include "stencil-composition/caches/cache.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"

namespace gridtools {

#define _DEFINE_CACHE(z, n, nil)                                                                                 \
    template < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename T) >                                               \
    BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n))< BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T) > define_caches( \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T)) {                                                              \
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <             \
                                                 BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T) >,                     \
                                    is_cache > ::value),                                                         \
            "argument provided to define_caches construct is not of the type cache");                            \
        return BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n))< BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T) >();  \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, _DEFINE_CACHE, _)
#undef _DEFINE_CACHE
} // namespace gridtools
