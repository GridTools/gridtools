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

#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {
    /**
     * @brief iterate domain class for the X86 backend
     */
    template <typename IterateDomainArguments>
    struct iterate_domain_x86 : iterate_domain<IterateDomainArguments> {
        using iterate_domain<IterateDomainArguments>::iterate_domain;

        template <class Arg, class Accessor>
        GT_FORCE_INLINE decltype(auto) deref(Accessor const &acc) const {
            return *this->template get_ptr<Arg>(acc);
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_x86<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
