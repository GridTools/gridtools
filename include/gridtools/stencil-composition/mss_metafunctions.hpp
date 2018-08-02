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
/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once

#include <tuple>

#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "./caches/cache_traits.hpp"
#include "./esf.hpp"

namespace gridtools {
    namespace _impl {
        template <class T>
        struct is_sequence_of_caches : is_sequence_of<T, is_cache> {};
    } // namespace _impl

    /**
     * @struct is_mss_parameter
     * metafunction that determines if a given type is a valid parameter for mss_descriptor
     */
    template <class T>
    struct is_mss_parameter : bool_constant<_impl::is_sequence_of_caches<T>::value || is_esf_descriptor<T>::value> {};

    /**
     * @struct extract_mss_caches
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all caches
     */
    template <typename... MssParameters>
    struct extract_mss_caches {
#ifdef __DISABLE_CACHING__
        typedef std::tuple<> type;
#else
        using tuple_of_caches = GT_META_CALL(
            meta::filter, (_impl::is_sequence_of_caches, std::tuple<MssParameters...>));

        GRIDTOOLS_STATIC_ASSERT(meta::length<tuple_of_caches>::value < 2,
            "Wrong number of sequence of caches. Probably caches are defined in multiple dinstinct instances of "
            "define_caches\n"
            "Only one instance of define_caches is allowed.");

        using type = typename conditional_t<meta::length<tuple_of_caches>::value == 0,
            meta::lazy::id<std::tuple<>>,
            meta::lazy::first<tuple_of_caches>>::type;
#endif
    };

    /**
     * @struct extract_mss_esfs
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all esf descriptors
     */
    template <class... Ts>
    GT_META_DEFINE_ALIAS(extract_mss_esfs, meta::filter, (is_esf_descriptor, std::tuple<Ts...>));

} // namespace gridtools
