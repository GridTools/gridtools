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

#include <type_traits>
#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the Host backend
     */
    template <typename IterateDomainArguments>
    struct iterate_domain_naive : iterate_domain<iterate_domain_naive<IterateDomainArguments>, IterateDomainArguments> {
        template <class Arg>
        GT_FORCE_INLINE iterate_domain_naive(Arg &&arg)
            : iterate_domain<iterate_domain_naive<IterateDomainArguments>, IterateDomainArguments>(
                  std::forward<Arg>(arg)) {}

        template <class Arg, class T>
        static GT_FORCE_INLINE auto deref_impl(T &&ptr) GT_AUTO_RETURN(*ptr);
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_naive<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
