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

#include "../common/hip_wrappers.hpp"
#ifdef __CUDACC__
#include "./apply_gpu.hpp"
#endif
#include "./apply.hpp"

#include "../communication/low_level/gcl_arch.hpp"
#include "./predicate.hpp"

/** \defgroup Boundary-Conditions Boundary Conditions
 */

namespace gridtools {

    namespace _impl {
        /** \ingroup Boundary-Conditions
         * @{
         */

        template <typename /*GclArch*/, typename BoundaryFunction, typename Predicate>
        struct select_apply {
            using type = boundary_apply<BoundaryFunction, Predicate>;
        };

#ifdef __CUDACC__
        template <typename BoundaryFunction, typename Predicate>
        struct select_apply<gcl_gpu, BoundaryFunction, Predicate> {
            using type = boundary_apply_gpu<BoundaryFunction, Predicate>;
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
       on runtime values (useful to deal with non-periodic distributed examples
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
            bc_apply.apply(make_target_view(data_fields)...);
        }
    };

    template <class Arch, class BoundaryFunction, class Predicate = default_predicate>
    auto make_boundary(
        array<halo_descriptor, 3> const &hd, BoundaryFunction &&boundary_f, Predicate &&predicate = Predicate()) {
        return boundary<BoundaryFunction, Arch, Predicate>(
            hd, std::forward<BoundaryFunction>(boundary_f), std::forward<Predicate>(predicate));
    }

    /** @} */

} // namespace gridtools
