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
    class iterate_domain_x86
        : public iterate_domain<iterate_domain_x86<IterateDomainArguments>, IterateDomainArguments> {
        using base_t = typename iterate_domain_x86::iterate_domain;
        using strides_cached_t = typename base_t::strides_cached_t;

      public:
        iterate_domain_x86(iterate_domain_x86 const &) = delete;
        iterate_domain_x86 &operator=(iterate_domain_x86 const &) = delete;

        template <class... Args>
        GT_FORCE_INLINE iterate_domain_x86(Args &&... args) : base_t(std::forward<Args>(args)...) {}

        GT_FORCE_INLINE strides_cached_t &strides_impl() { return m_strides; }
        GT_FORCE_INLINE strides_cached_t const &strides_impl() const { return m_strides; }

        template <class Arg, class Ptr>
        static GT_FORCE_INLINE auto deref_impl(Ptr &&ptr) GT_AUTO_RETURN(*ptr);

      private:
        strides_cached_t m_strides;
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_x86<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
