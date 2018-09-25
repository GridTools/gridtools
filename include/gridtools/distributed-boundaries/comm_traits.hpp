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

#include "../common/boollist.hpp"
#ifdef _GCL_MPI_
#include "../communication/low-level/proc_grids_3D.hpp"
#else
#include "./mock_pattern.hpp"
#endif

#include "../stencil-composition/stencil-composition.hpp"

namespace gridtools {

#ifndef _GCL_MPI_
    using namespace mock_;
#endif

    /** \ingroup Distributed-Boundaries
     * @{ */

    template <typename StorageType, typename Arch>
    struct comm_traits {
        template <typename GCLArch, typename = void>
        struct compute_arch_of {
            using type = platform::x86;
        };

        template <typename T>
        struct compute_arch_of<gcl_gpu, T> {
            using type = platform::cuda;
        };

        using proc_layout = gridtools::layout_map<0, 1, 2>;
        using proc_grid_type = gridtools::MPI_3D_process_grid_t<3>;
        using comm_arch_type = Arch;
        using compute_arch = typename compute_arch_of<comm_arch_type>::type;
        static constexpr int version = gridtools::version_manual;
        using data_layout = typename StorageType::storage_info_t::layout_t;
        using value_type = typename StorageType::data_t;
    };

    /** @} */

} // namespace gridtools
