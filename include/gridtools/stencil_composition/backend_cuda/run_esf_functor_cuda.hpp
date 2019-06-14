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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"

namespace gridtools {
    namespace cuda {
        namespace _impl {
            GT_FUNCTION_DEVICE void final_sync(std::true_type) { __syncthreads(); }
            GT_FUNCTION_DEVICE void final_sync(std::false_type) {}
        } // namespace _impl

        template <class StageGroups, class Domain>
        GT_FUNCTION_DEVICE void run_esf_functor_cuda(Domain &domain) {

            GT_STATIC_ASSERT(!meta::is_empty<StageGroups>::value, GT_INTERNAL_ERROR);
            using first_t = meta::first<StageGroups>;
            using rest_t = meta::pop_front<StageGroups>;

            auto exec_stage_group = [&](auto stages) {
                device::for_each<decltype(stages)>([&](auto stage) {
                    if (domain.template is_thread_in_domain<typename decltype(stage)::extent_t>())
                        stage.exec(domain);
                });
            };

            // execute the groups of independent stages calling `__syncthreads()` in between
            exec_stage_group(first_t{});
            device::for_each<rest_t>([&](auto stages) {
                __syncthreads();
                exec_stage_group(stages);
            });

            // call additional `__syncthreads()` at the end of the k-level if the domain has IJ caches
            _impl::final_sync(domain.has_ij_caches());
        }
    } // namespace cuda
} // namespace gridtools
