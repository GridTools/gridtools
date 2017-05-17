/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include "cache.hpp"

namespace gridtools {

    /**
     * @struct is_cache
     * metafunction determining if a type is a cache type
     */
    template < typename T >
    struct is_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_cache< detail::cache_impl< cacheType, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct is_ij_cache
     * metafunction determining if a type is a cache of IJ type
     */
    template < typename T >
    struct is_ij_cache : boost::mpl::false_ {};

    template < typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_ij_cache< detail::cache_impl< IJ, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct is_k_cache
     * metafunction determining if a type is a cache of K type
     */
    template < typename T >
    struct is_k_cache : boost::mpl::false_ {};

    template < typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_k_cache< detail::cache_impl< K, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct is_flushing_cache
     * metafunction determining if a type is a flush cache
     */
    template < typename T >
    struct is_flushing_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, typename Interval >
    struct is_flushing_cache< detail::cache_impl< cacheType, Arg, cache_io_policy::flush, Interval > >
        : boost::mpl::true_ {};

    /**
     * @struct is_epflushing_cache
     * metafunction determining if a type is an end-point flushing cache
     */
    template < typename T >
    struct is_epflushing_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, typename Interval >
    struct is_epflushing_cache< detail::cache_impl< cacheType, Arg, cache_io_policy::epflush, Interval > >
        : boost::mpl::true_ {};

    /**
     * @struct is_filling_cache
     * metafunction determining if a type is a filling cache
     */
    template < typename T >
    struct is_filling_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, typename Interval >
    struct is_filling_cache< detail::cache_impl< cacheType, Arg, cache_io_policy::fill, Interval > >
        : boost::mpl::true_ {};

    /**
     * @struct is_bpfilling_cache
     * metafunction determining if a type is a begin-point cache
     */
    template < typename T >
    struct is_bpfilling_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, typename Interval >
    struct is_bpfilling_cache< detail::cache_impl< cacheType, Arg, cache_io_policy::bpfill, Interval > >
        : boost::mpl::true_ {};

    /**
     * @struct cache_parameter
     *  trait returning the parameter Arg type of a user provided cache
     */
    template < typename T >
    struct cache_parameter;

    template < cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct cache_parameter< detail::cache_impl< cacheType, Arg, cacheIOPolicy, Interval > > {
        typedef Arg type;
    };

} // namespace gridtools
