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
#ifdef __CUDACC__
#include "../storage/storage_cuda/data_view_helpers.hpp"
#include "./apply_gpu.hpp"
#endif
#include "../storage/storage_host/data_view_helpers.hpp"
#include "../storage/storage_mc/data_view_helpers.hpp"
#include "./apply.hpp"

#include "./predicate.hpp"

/** \defgroup Boundary-Conditions Boundary Conditions
 */

namespace gridtools {

    namespace _impl {
        /** \ingroup Boundary-Conditions
         * @{
         */

        template <class Arch, typename BoundaryFunction, typename Predicate>
        struct select_apply;

        template <typename BoundaryFunction, typename Predicate>
        struct select_apply<target::x86, BoundaryFunction, Predicate> {
            using type = boundary_apply<BoundaryFunction, Predicate>;
        };

        template <typename BoundaryFunction, typename Predicate>
        struct select_apply<target::mc, BoundaryFunction, Predicate> {
            using type = boundary_apply<BoundaryFunction, Predicate>;
        };

#ifdef __CUDACC__
        template <typename BoundaryFunction, typename Predicate>
        struct select_apply<target::cuda, BoundaryFunction, Predicate>

        {
            using type = boundary_apply_gpu<BoundaryFunction, Predicate>;
        };
#endif

        template <class Arch, access_mode AM, typename DataF>
        struct proper_view;

        template <access_mode AM, typename DataF>
        struct proper_view<target::x86, AM, DataF> {
            using proper_view_t = decltype(make_host_view<AM, DataF>(std::declval<DataF>()));

            static proper_view_t make(DataF const &df) { return make_host_view<AM>(df); }
        };

        template <access_mode AM, typename DataF>
        struct proper_view<target::mc, AM, DataF> {
            using proper_view_t = decltype(make_host_view<AM, DataF>(std::declval<DataF>()));

            static proper_view_t make(DataF const &df) { return make_host_view<AM>(df); }
        };

#ifdef __CUDACC__
        template <access_mode AM, typename DataF>
        struct proper_view<target::cuda, AM, DataF> {
            using proper_view_t = decltype(make_device_view<AM, DataF>(std::declval<DataF>()));

            static proper_view_t make(DataF const &df) { return make_device_view<AM>(df); }
        };
#endif
        /** @} */
    } // namespace _impl

    /** \ingroup Boundary-Conditions
     * @{
     */

    /**
       @brief Main interface for boundary condition application.

       \tparam BoundaryFunction The boundary condition functor
       \tparam Arch The target where the data is (e.g., Host or Cuda)
       \tparam Predicate Runtime predicate for deciding if to apply boundary conditions or not on certain regions based
       on runtime values (useful to deal with non-priodic distributed examples
     */
    template <typename BoundaryFunction, class Arch, typename Predicate = default_predicate>
    struct boundary {
        using bc_apply_t = typename _impl::select_apply<Arch, BoundaryFunction, Predicate>::type;

        bc_apply_t bc_apply;

        boundary(
            array<halo_descriptor, 3> const &hd, BoundaryFunction const &boundary_f, Predicate predicate = Predicate())
            : bc_apply(hd, boundary_f, predicate) {}

        template <typename... DataFields>
        void apply(DataFields &... data_fields) const {
            bc_apply.apply(
                _impl::proper_view<Arch, access_mode::read_write, typename std::decay<DataFields>::type>::make(
                    data_fields)...);
        }
    };

    /** @} */

} // namespace gridtools
