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

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "./execution_types.hpp"
#include "./iterate_domain_fwd.hpp"
#include "./level.hpp"

namespace gridtools {

    /**\brief policy defining the behaviour on the vertical direction*/
    template <class From, class To, class ExecutionType>
    struct iteration_policy {
        GT_STATIC_ASSERT(is_level<From>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_level<To>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(level_to_index<From>::value <= level_to_index<From>::value, GT_INTERNAL_ERROR);

        using from = From;
        using to = To;

        using execution_type = ExecutionType;

        GT_FUNCTION static int_t increment(int_t &k) { return ++k; }
        GT_FUNCTION static int_t decrement(int_t &k) { return --k; }

        template <typename IterateDomain>
        GT_FUNCTION static void increment(IterateDomain &eval) {
            GT_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k();
        }

        template <typename IterateDomain>
        GT_FUNCTION static void increment_by(IterateDomain &eval, int_t step) {
            GT_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k(step);
        }

        GT_FUNCTION static bool condition(int_t a, int_t b) { return a <= b; }
    };

    /**\brief specialization for the backward iteration loop over k*/
    template <class From, class To>
    struct iteration_policy<From, To, execute::backward> {
        GT_STATIC_ASSERT(is_level<From>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_level<To>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(level_to_index<From>::value >= level_to_index<From>::value, GT_INTERNAL_ERROR);

        using from = From;
        using to = To;

        using execution_type = execute::backward;

        GT_FUNCTION static int_t increment(int_t &k) { return --k; }
        GT_FUNCTION static int_t decrement(int_t &k) { return ++k; }

        template <typename Domain>
        GT_FUNCTION static void increment(Domain &dom) {
            GT_STATIC_ASSERT(is_iterate_domain<Domain>::value, GT_INTERNAL_ERROR);
            using namespace literals;
            dom.template increment_k(-1_c);
        }

        template <typename IterateDomain>
        GT_FUNCTION static void increment_by(IterateDomain &eval, int_t step) {
            GT_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k(-step);
        }

        GT_FUNCTION static bool condition(int_t const &a, int_t const &b) { return a >= b; }
    };

    template <typename T>
    struct is_iteration_policy : std::false_type {};

    template <typename From, typename To, typename ExecutionType>
    struct is_iteration_policy<iteration_policy<From, To, ExecutionType>> : std::true_type {};

} // namespace gridtools
