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
    using esf_param_list = typename Esf::esf_function_t::param_list;

    namespace lazy {
        template <class Esf, class Args>
        struct esf_replace_args;
        template <class F, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor<F, OldArgs>, NewArgs> {
            using type = esf_descriptor<F, NewArgs>;
        };
    } // namespace lazy
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
