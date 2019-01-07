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

#include "../meta/dedup.hpp"
#include "../meta/defs.hpp"
#include "../meta/filter.hpp"
#include "../meta/is_empty.hpp"
#include "../meta/length.hpp"
#include "../meta/logical.hpp"
#include "../meta/macros.hpp"
#include "../meta/transform.hpp"

namespace gridtools {

    namespace _impl {
#if GT_BROKEN_TEMPLATE_ALIASES
        template <class Stage>
        struct get_extent_from_stage {
            using type = typename Stage::extent_t;
        };
#else
        template <class Stage>
        using get_extent_from_stage = typename Stage::extent_t;
#endif
        template <class Extent>
        struct has_same_extent {
            template <class Stage>
            GT_META_DEFINE_ALIAS(apply, std::is_same, (Extent, typename Stage::extent_t));
        };

        template <class AllStages>
        struct stages_with_the_given_extent {
            template <class Extent>
            GT_META_DEFINE_ALIAS(apply, meta::filter, (has_same_extent<Extent>::template apply, AllStages));
        };

        GT_META_LAZY_NAMESPACE {
            template <template <class...> class CompoundStage, class Stages>
            struct fuse_stages_with_the_same_extent;
            template <template <class...> class CompoundStage, template <class...> class L, class... Stages>
            struct fuse_stages_with_the_same_extent<CompoundStage, L<Stages...>> {
                using type = CompoundStage<Stages...>;
            };
            template <template <class...> class CompoundStage, template <class...> class L, class Stage>
            struct fuse_stages_with_the_same_extent<CompoundStage, L<Stage>> {
                using type = Stage;
            };
        }
        GT_META_DELEGATE_TO_LAZY(fuse_stages_with_the_same_extent,
            (template <class...> class CompoundStage, class Stages),
            (CompoundStage, Stages));

        template <template <class...> class CompoundStage>
        struct fuse_stages_with_the_same_extent_f {
            template <class Stages>
            GT_META_DEFINE_ALIAS(apply, fuse_stages_with_the_same_extent, (CompoundStage, Stages));
        };

    } // namespace _impl

    GT_META_LAZY_NAMESPACE {
        template <template <class...> class CompoundStage, class Stages>
        struct fuse_stages {
            GRIDTOOLS_STATIC_ASSERT(meta::length<Stages>::value > 1, GT_INTERNAL_ERROR);
            using all_extents_t = GT_META_CALL(meta::transform, (_impl::get_extent_from_stage, Stages));
            using extents_t = GT_META_CALL(meta::dedup, all_extents_t);
            GRIDTOOLS_STATIC_ASSERT(!meta::is_empty<extents_t>::value, GT_INTERNAL_ERROR);
            using stages_grouped_by_extent_t = GT_META_CALL(
                meta::transform, (_impl::stages_with_the_given_extent<Stages>::template apply, extents_t));
            GRIDTOOLS_STATIC_ASSERT(
                (!meta::any_of<meta::is_empty, stages_grouped_by_extent_t>::value), GT_INTERNAL_ERROR);
            using type = GT_META_CALL(meta::transform,
                (_impl::fuse_stages_with_the_same_extent_f<CompoundStage>::template apply, stages_grouped_by_extent_t));
        };

        template <template <class...> class CompoundStage, template <class...> class L, class Stage>
        struct fuse_stages<CompoundStage, L<Stage>> {
            using type = L<Stage>;
        };
        template <template <class...> class CompoundStage, template <class...> class L>
        struct fuse_stages<CompoundStage, L<>> {
            using type = L<>;
        };
    }
    /**
     *  Group the stages from the input by extent and substitute each group by compound stage.
     */
    GT_META_DELEGATE_TO_LAZY(
        fuse_stages, (template <class...> class CompoundStage, class Stages), (CompoundStage, Stages));

} // namespace gridtools