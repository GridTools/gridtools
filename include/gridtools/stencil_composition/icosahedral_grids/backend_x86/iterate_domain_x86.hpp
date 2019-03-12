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
    class iterate_domain_x86
        : public iterate_domain<iterate_domain_x86<IterateDomainArguments>, IterateDomainArguments> {
        using base_t = iterate_domain<iterate_domain_x86<IterateDomainArguments>, IterateDomainArguments>;
        using strides_cached_t = typename base_t::strides_cached_t;

        strides_cached_t m_strides;

      public:
        template <class Arg>
        GT_FORCE_INLINE iterate_domain_x86(Arg &&arg) : base_t(std::forward<Arg>(arg)) {}

        GT_FORCE_INLINE strides_cached_t &strides_impl() { return m_strides; }
        GT_FORCE_INLINE strides_cached_t const &strides_impl() const { return m_strides; }

        template <class Arg, class T>
        static GT_FORCE_INLINE auto deref_impl(T &&ptr) GT_AUTO_RETURN(*ptr);
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_x86<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
