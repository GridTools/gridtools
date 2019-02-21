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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/first.hpp"
#include "../../meta/is_empty.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/pop_front.hpp"

namespace gridtools {
    namespace _impl {
        template <class ItDomain>
        struct exec_stage_f {
            ItDomain &m_domain;
            template <class Stage>
            GT_FUNCTION void operator()() const {
                if (m_domain.template is_thread_in_domain<typename Stage::extent_t>())
                    Stage::exec(m_domain);
            }
        };

        template <class Stages, class ItDomain>
        GT_FUNCTION void exec_stage_group(ItDomain &it_domain) {
            host_device::for_each_type<Stages>(exec_stage_f<ItDomain>{it_domain});
        }

        template <class ItDomain>
        struct exec_stage_group_f {
            ItDomain &m_domain;

            template <class Stages>
            GT_FUNCTION void operator()() const {
#ifdef __CUDA_ARCH__
                __syncthreads();
#endif
                exec_stage_group<Stages>(m_domain);
            }
        };
    } // namespace _impl

    struct run_esf_functor_cuda {
        template <class StageGroups, class ItDomain>
        GT_FUNCTION static void exec(ItDomain &it_domain) {

            GT_STATIC_ASSERT(!meta::is_empty<StageGroups>::value, GT_INTERNAL_ERROR);
            using first_t = GT_META_CALL(meta::first, StageGroups);
            using rest_t = GT_META_CALL(meta::pop_front, StageGroups);

            // execute the groups of independent stages calling `__syncthreads()` in between
            _impl::exec_stage_group<first_t>(it_domain);
            host_device::for_each_type<rest_t>(_impl::exec_stage_group_f<ItDomain>{it_domain});

            // call additional `__syncthreads()` at the end of the k-level if the domain has IJ caches
#ifdef __CUDA_ARCH__
            if (ItDomain::has_ij_caches)
                __syncthreads();
#endif
        }
    };
} // namespace gridtools
