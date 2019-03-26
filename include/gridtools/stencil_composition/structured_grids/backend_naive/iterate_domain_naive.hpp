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
    class iterate_domain_naive
        : public iterate_domain<iterate_domain_naive<IterateDomainArguments>, IterateDomainArguments> {
        using base_t = typename iterate_domain_naive::iterate_domain;

      public:
        iterate_domain_naive(iterate_domain_naive const &) = delete;
        iterate_domain_naive &operator=(iterate_domain_naive const &) = delete;

        template <class... Args>
        GT_FORCE_INLINE iterate_domain_naive(Args &&... args) : base_t(std::forward<Args>(args)...) {}

        template <class Arg, class Ptr>
        static GT_FORCE_INLINE auto deref_impl(Ptr &&ptr) GT_AUTO_RETURN(*ptr);
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_naive<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
