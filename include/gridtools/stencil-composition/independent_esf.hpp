/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../common/defs.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../meta/logical.hpp"
#include "./esf_fwd.hpp"

namespace gridtools {

    template <class>
    struct independent_esf;

    template <class T>
    GT_META_DEFINE_ALIAS(is_independent, meta::is_instantiation_of, (independent_esf, T));

    template <class Esfs>
    struct is_esf_descriptor<independent_esf<Esfs>> : std::true_type {};

    template <class Esfs>
    struct independent_esf {
        GT_STATIC_ASSERT(
            (meta::all_of<is_esf_descriptor, Esfs>::value), "Error: independent_esf requires a sequence of esf's");
        // independent_esf always contains a flat list of esfs! No independent_esf inside.
        // This is ensured by make_independent design. That's why this assert is internal.
        GT_STATIC_ASSERT((!meta::any_of<is_independent, Esfs>::value), GT_INTERNAL_ERROR);
        using esf_list = Esfs;
    };

} // namespace gridtools
