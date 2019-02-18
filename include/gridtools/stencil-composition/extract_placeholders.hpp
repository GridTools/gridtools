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

#include "../meta.hpp"
#include "esf_metafunctions.hpp"

namespace gridtools {
    namespace extract_placeholders_impl_ {
        template <class Trees, template <class...> class FlattenTree>
        GT_META_DEFINE_ALIAS(flatten_trees, meta::flatten, (GT_META_CALL(meta::transform, (FlattenTree, Trees))));

        // Extract args from ESF.
        template <class Esf>
        GT_META_DEFINE_ALIAS(get_args, meta::id, typename Esf::args_t);

        // Extract ESFs from an MSS.
        template <class Mss>
        GT_META_DEFINE_ALIAS(get_esfs, unwrap_independent, typename Mss::esf_sequence_t);
    } // namespace extract_placeholders_impl_

    template <class Mss, class Esfs = GT_META_CALL(unwrap_independent, typename Mss::esf_sequence_t)>
    GT_META_DEFINE_ALIAS(extract_placeholders_from_mss,
        meta::dedup,
        (GT_META_CALL(extract_placeholders_impl_::flatten_trees, (Esfs, extract_placeholders_impl_::get_args))));

    /// Takes a type list of MSS descriptions and returns deduplicated type list of all placeholders
    /// that are used in the given msses.
    template <class Msses>
    GT_META_DEFINE_ALIAS(extract_placeholders_from_msses,
        meta::dedup,
        (GT_META_CALL(extract_placeholders_impl_::flatten_trees,
            (GT_META_CALL(extract_placeholders_impl_::flatten_trees, (Msses, extract_placeholders_impl_::get_esfs)),
                extract_placeholders_impl_::get_args))));
} // namespace gridtools
