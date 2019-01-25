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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/copy_into_variadic.hpp"
#include "../../meta.hpp"
#include "../esf_metafunctions.hpp"
#include "../extent.hpp"
#include "./cache_traits.hpp"

namespace gridtools {
    namespace extract_extent_caches_impl_ {
        template <class Arg>
        struct arg_extent_from_esf {

            template <class EsfArg, class Accessor>
            GT_META_DEFINE_ALIAS(
                get_extent, meta::if_, (std::is_same<Arg, EsfArg>, typename Accessor::extent_t, extent<>));

            template <class Esf,
                class Args = typename Esf::args_t,
                class Accessors = copy_into_variadic<typename esf_arg_list<Esf>::type, meta::list<>>,
                class Extents = GT_META_CALL(meta::transform, (get_extent, Args, Accessors))>
            GT_META_DEFINE_ALIAS(apply, meta::rename, (enclosing_extent, Extents));
        };
    } // namespace extract_extent_caches_impl_

    template <class Arg,
        class Esfs,
        class Extents = GT_META_CALL(
            meta::transform, (extract_extent_caches_impl_::arg_extent_from_esf<Arg>::template apply, Esfs))>
    GT_META_DEFINE_ALIAS(extract_k_extent_for_cache, meta::rename, (enclosing_extent, Extents));

} // namespace gridtools
