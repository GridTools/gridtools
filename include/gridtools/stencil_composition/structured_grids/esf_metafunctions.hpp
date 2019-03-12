/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/defs.hpp"
#include "../../meta/macros.hpp"
#include "./esf.hpp"

namespace gridtools {

    template <class Esf>
    GT_META_DEFINE_ALIAS(esf_param_list, meta::id, typename Esf::esf_function_t::param_list);

    /** Retrieve the extent in esf_descriptor_with_extents

       \tparam Esf The esf_descriptor that must be the one speficying the extent
    */
    template <typename Esf>
    struct esf_extent;

    template <typename ESF, typename Extent, typename ArgArray>
    struct esf_extent<esf_descriptor_with_extent<ESF, Extent, ArgArray>> {
        using type = Extent;
    };

    GT_META_LAZY_NAMESPACE {
        template <class Esf, class Args>
        struct esf_replace_args;
        template <class F, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor<F, OldArgs>, NewArgs> {
            using type = esf_descriptor<F, NewArgs>;
        };
        template <class F, class Extent, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor_with_extent<F, Extent, OldArgs>, NewArgs> {
            using type = esf_descriptor_with_extent<F, Extent, NewArgs>;
        };
    }
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
