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
#pragma once
#include <boost/mpl/assert.hpp>
#include <boost/mpl/map/map0.hpp>
#include <boost/mpl/transform.hpp>

#include "caches/cache_metafunctions.hpp"
#include "caches/cache_traits.hpp"
#include "esf.hpp"
#include "functor_do_methods.hpp"
#include "independent_esf.hpp"

// #include "stencil-composition/sfinae.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    namespace _impl {

        /**@brief Macro defining a sfinae metafunction

           defines a metafunction has_extent_type, which returns true if its template argument
           defines a type called extent_type. It also defines a get_extent_type metafunction, which
           can be used to return the extent_type only when it is present, without giving compilation
           errors in case it is not defined.
         */
        // HAS_TYPE_SFINAE(extent_type, has_extent_type, get_extent_type)
    }

    template < typename Mss1, typename Mss2, typename Tag >
    struct condition;

    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence = boost::mpl::vector0<> >
    struct mss_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfDescrSequence, is_esf_descriptor >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_execution_engine< ExecutionEngine >::value), GT_INTERNAL_ERROR);
        typedef ExecutionEngine execution_engine_t;
        typedef EsfDescrSequence esf_sequence_t;
        typedef CacheSequence cache_sequence_t;
        typedef static_bool< false > is_reduction_t;
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

    template < typename Mss >
    struct mss_descriptor_cache_sequence {};

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_cache_sequence< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef CacheSequence type;
    };

    template < typename Mss >
    struct mss_descriptor_is_reduction;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_is_reduction< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef static_bool< false > type;
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
