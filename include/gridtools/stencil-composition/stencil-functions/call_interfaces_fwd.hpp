/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

namespace gridtools {
    namespace _impl {
        template <typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedAccessors,
            typename ReturnType,
            int OutArg>
        struct function_aggregator;

        template <typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedArguments>
        struct function_aggregator_procedure;

        template <typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedArguments>
        struct function_aggregator_procedure_offsets;
    } // namespace _impl
} // namespace gridtools
