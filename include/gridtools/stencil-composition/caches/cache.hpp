/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
/**
   @file
   @brief File containing the definition of caches. They are the API exposed to the user to describe
   parameters that will be cached in a on-chip memory.
*/

#pragma once

#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor.hpp>
#include <boost/type_traits/is_same.hpp>

#include <common/defs.hpp>
#include <common/generic_metafunctions/mpl_vector_flatten.hpp>
#include <common/generic_metafunctions/variadic_to_vector.hpp>
#include <common/gt_assert.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/caches/cache_definitions.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/location_type.hpp>

namespace gridtools {

    template < int M, int P >
    struct window {
        static constexpr int m_ = M;
        static constexpr int p_ = P;
        GRIDTOOLS_STATIC_ASSERT((m_ == 0 || p_ == 0),
            "One of the bounds of the cache window has to be 0, (upper bound for a forward loop or lower bound for a "
            "backward loop) since it does not make sense to flush the head of the cache nor fill the tail.");
    };

    template < typename T >
    struct is_window : boost::mpl::false_ {};

    template < int M, int P >
    struct is_window< window< M, P > > : boost::mpl::true_ {};

    template < cache_io_policy CacheIOPolicy, typename T >
    struct window_get_size;

    template < cache_io_policy CacheIOPolicy, int M, int P >
    struct window_get_size< CacheIOPolicy, window< M, P > > {
        using type = static_int< P - M + 1 + ((CacheIOPolicy == cache_io_policy::epflush) ? (-1) : 0) >;
    };

    template < cache_io_policy CacheIOPolicy, typename IterationPolicy, typename T >
    struct window_get_min;

    template < cache_io_policy CacheIOPolicy, typename IterationPolicy, int M, int P >
    struct window_get_min< CacheIOPolicy, IterationPolicy, window< M, P > > {
        using type = static_int<
            (IterationPolicy::value == enumtype::forward && CacheIOPolicy == cache_io_policy::epflush) ? M + 1 : M >;
    };

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
         * @tparam Interval vertical interval of validity of the cache
         */
        template < cache_type cacheType,
            typename Arg,
            cache_io_policy cacheIOPolicy,
            typename Interval,
            typename KWindow >
        struct cache_impl {
            GRIDTOOLS_STATIC_ASSERT(
                (is_arg< Arg >::value), "argument passed to ij cache is not of the right arg<> type");
            typedef Arg arg_t;
// TODO ICO_STORAGE
#ifndef STRUCTURED_GRIDS
            GRIDTOOLS_STATIC_ASSERT(
                (!boost::is_same< typename Arg::location_t, enumtype::default_location_type >::value),
                "args in irregular grids require a location type");
#endif
            typedef Interval interval_t;
            using kwindow_t = KWindow;
            typedef enumtype::enum_type< cache_type, cacheType > cache_type_t;
            static constexpr cache_io_policy ccacheIOPolicy = cacheIOPolicy;
        };

        /**
        * @brief helper metafunction class that is used to force the resolution of an mpl placeholder type
        */
        template < cache_type cacheType, cache_io_policy cacheIOPolicy, typename Interval, typename KWindow >
        struct force_arg_resolution {
            template < typename T >
            struct apply {
                typedef cache_impl< cacheType, T, cacheIOPolicy, Interval, KWindow > type;
            };
        };
    }

    /**
     *	@brief function that forms a vector of caches that share the same cache type and input/output policy  (c++11
     *version)
     *	@tparam cacheType type of cache (e.g., IJ, IJK, ...)
     *	@tparam cacheIOPolicy input/output policy (e.g., cFill, cLocal, ...)
     *	@tparam Args arbitrary number of storages that should be cached
     *	@return vector of caches
     */
    template < cache_type cacheType,
        cache_io_policy cacheIOPolicy,
        typename Interval = boost::mpl::void_,
        typename KWindow = boost::mpl::void_,
        typename... Args >
    constexpr typename boost::mpl::transform< boost::mpl::vector< Args... >,
        detail::force_arg_resolution< cacheType, cacheIOPolicy, Interval, KWindow > >::type
    cache(Args &&...) {
        GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) > 0, "Cannot build cache sequence without argument");
        GRIDTOOLS_STATIC_ASSERT(((boost::is_same< Interval, boost::mpl::void_ >::value) || cacheType == K),
            "Passing an interval to the cache<> construct is only allowed and required by the K caches");
        GRIDTOOLS_STATIC_ASSERT((!(boost::is_same< Interval, boost::mpl::void_ >::value) || cacheType != K ||
                                    cacheIOPolicy == cache_io_policy::local),
            "cache<K, ... > construct requires an interval (unless the IO policy is local)");

        GRIDTOOLS_STATIC_ASSERT((!(boost::is_same< KWindow, boost::mpl::void_ >::value) ||
                                    !(cacheType == K && (cacheIOPolicy == cache_io_policy::bpfill ||
                                                            cacheIOPolicy == cache_io_policy::epflush))),
            "cache<K, ... > construct requires a k window for bpfill and epflush");

        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same< Interval, boost::mpl::void_ >::value || is_interval< Interval >::value),
            "Invalid Interval type passed to cache construct");
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< KWindow, boost::mpl::void_ >::value || is_window< KWindow >::value),
            "Invalid k-window type passed to cache construct");

        typedef typename boost::mpl::transform< boost::mpl::vector< Args... >,
            detail::force_arg_resolution< cacheType, cacheIOPolicy, Interval, KWindow > >::type res_ty;
        return res_ty();
    }
}
