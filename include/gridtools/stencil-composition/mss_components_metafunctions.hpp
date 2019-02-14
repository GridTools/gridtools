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

#include <tuple>

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "esf_metafunctions.hpp"
#include "mss.hpp"
#include "mss_components.hpp"

namespace gridtools {
    namespace mss_comonents_metafunctions_impl_ {
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct mss_split_esfs;
            template <class ExecutionEngine, class EsfSequence, class CacheSequence>
            struct mss_split_esfs<mss_descriptor<ExecutionEngine, EsfSequence, CacheSequence>> {
                GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
                template <class Esf>
                GT_META_DEFINE_ALIAS(
                    make_mss, meta::id, (mss_descriptor<ExecutionEngine, std::tuple<Esf>, std::tuple<>>));
                using esfs_t = GT_META_CALL(unwrap_independent, EsfSequence);
                using type = GT_META_CALL(meta::transform, (make_mss, esfs_t));
            };
        }
        GT_META_DELEGATE_TO_LAZY(mss_split_esfs, class Mss, Mss);

        template <bool Fuse, class Msses>
        struct split_mss_into_independent_esfs {
            using mms_lists_t = GT_META_CALL(meta::transform, (mss_split_esfs, Msses));
            using type = GT_META_CALL(meta::flatten, mms_lists_t);
        };

        template <class Msses>
        struct split_mss_into_independent_esfs<true, Msses> {
            using type = Msses;
        };

        template <class ExtentMap, class Axis>
        struct make_mms_components_f {
            template <class Mss>
            GT_META_DEFINE_ALIAS(apply, meta::id, (mss_components<Mss, ExtentMap, Axis>));
        };

    } // namespace mss_comonents_metafunctions_impl_

    /**
     * @brief metafunction that builds the array of mss components
     */
    template <bool Fuse,
        class Msses,
        class ExtentMap,
        class Axis,
        class SplitMsses =
            typename mss_comonents_metafunctions_impl_::split_mss_into_independent_esfs<Fuse, Msses>::type,
        class Maker = mss_comonents_metafunctions_impl_::make_mms_components_f<ExtentMap, Axis>>
    GT_META_DEFINE_ALIAS(build_mss_components_array, meta::transform, (Maker::template apply, SplitMsses));

} // namespace gridtools
