/**
   @file
   @brief File containing the definition of caches. They are the API exposed to the user to describe
   parameters that will be cached in a on-chip memory.
*/

#pragma once

#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/size.hpp>
#include <boost/preprocessor.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/mpl_vector_flatten.hpp"
#include "../../stencil-composition/caches/cache_definitions.hpp"
#include "../../stencil-composition/accessor.hpp"

namespace gridtools {

    namespace detail {
        /**
         * @struct cache_impl
         * The cache type is described with a template parameter to the class
         * Caching assumes a parallelization model where all the processing all elements in the vertical dimension are
         * private to each parallel thread,
         * while the processing of grid points in the horizontal plane is executed by different parallel threads.
         * Those caches that cover data in the horizontal (IJ and IJK) are accessed by parallel core units, and
         * therefore require synchronization capabilities (for example shared memory in the GPU), like IJ or IJK caches.
         * On the contrary caches in the K dimension are only accessed by one thread, and therefore resources can be
         * allocated
         * in on-chip without synchronization capabilities (for example registers in GPU)
         * @tparam  cacheType type of cache
         * @tparam Arg argument with parameter being cached
         * @tparam CacheIOPolicy IO policy for cache
         */
        template < cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy >
        struct cache_impl {
            GRIDTOOLS_STATIC_ASSERT(
                (is_arg< Arg >::value), "argument passed to ij cache is not of the right arg<> type");
            typedef Arg arg_t;
            typedef enumtype::enum_type< cache_type, cacheType > cache_type_t;
        };

        /**
        * @brief helper metafunction class that is used to force the resolution of an mpl placeholder type
        */
        template < cache_type cacheType, cache_io_policy cacheIOPolicy >
        struct force_arg_resolution {
            template < typename T >
            struct apply {
                typedef cache_impl< cacheType, T, cacheIOPolicy > type;
            };
        };
    }

#ifdef CXX11_ENABLED
    /**
     *	@brief function that forms a vector of caches that share the same cache type and input/output policy  (c++11
     *version)
     *	@tparam cacheType type of cache (e.g., IJ, IJK, ...)
     *	@tparam cacheIOPolicy input/output policy (e.g., cFill, cLocal, ...)
     *	@tparam Args arbitrary number of storages that should be cached
     *	@return vector of caches
     */
    template < cache_type cacheType, cache_io_policy cacheIOPolicy, typename... Args >
    constexpr typename boost::mpl::transform< boost::mpl::vector< Args... >,
        detail::force_arg_resolution< cacheType, cacheIOPolicy > >::type
    cache(Args &&...) {
        GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) > 0, "Cannot build cache sequence without argument");
        typedef typename boost::mpl::transform< boost::mpl::vector< Args... >,
            detail::force_arg_resolution< cacheType, cacheIOPolicy > >::type res_ty;
        return res_ty();
    }
#else

/*
 * This macro is providing the same functionality as the cache(Args&&) function above.
 * Just used because of c++03 compatibility.
 */
#define _CREATE_CACHE(z, n, nil)                                                                                     \
    template < cache_type cacheType,                                                                                 \
        cache_io_policy cacheIOPolicy,                                                                               \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename T) >                                                          \
    typename boost::mpl::transform< boost::mpl::vector< BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T) >,                  \
        detail::force_arg_resolution< cacheType, cacheIOPolicy > >::type cache(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), \
        T)) {                                                                                                        \
        typedef typename boost::mpl::transform< boost::mpl::vector< BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), T) >,      \
            detail::force_arg_resolution< cacheType, cacheIOPolicy > >::type res_type;                               \
        GRIDTOOLS_STATIC_ASSERT(                                                                                     \
            (boost::mpl::size< res_type >::value > 0), "Cannot build cache sequence without argument");              \
        return res_type();                                                                                           \
    }
    BOOST_PP_REPEAT(GT_MAX_ARGS, _CREATE_CACHE, _)
#undef _CREATE_CACHE

#endif
}
