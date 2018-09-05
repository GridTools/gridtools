/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/host_device.hpp"

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
            for_each_type<Stages>(exec_stage_f<ItDomain>{it_domain});
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

            GRIDTOOLS_STATIC_ASSERT(!meta::is_empty<StageGroups>::value, GT_INTERNAL_ERROR);
            using first_t = GT_META_CALL(meta::first, StageGroups);
            using rest_t = GT_META_CALL(meta::pop_front, StageGroups);

            // execute the groups of independent stages calling `__syncthreads()` in between
            _impl::exec_stage_group<first_t>(it_domain);
            for_each_type<rest_t>(_impl::exec_stage_group_f<ItDomain>{it_domain});

            // call additional `__syncthreads()` at the end of the k-level if the domain has IJ caches
#ifdef __CUDA_ARCH__
            if (ItDomain::has_ij_caches)
                __syncthreads();
#endif
        }
    };
} // namespace gridtools
