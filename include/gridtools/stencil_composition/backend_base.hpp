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

#include <cassert>
#include <type_traits>

#include "../common/defs.hpp"
#include "../common/selector.hpp"
#include "../storage/storage_facility.hpp"

/**
   @file
   @brief base class for all the backends. Current supported backend are ::gridtools::target::x86 and
   ::gridtools::target::cuda
*/

namespace gridtools {
    /**
        this struct contains the 'run' method for all backends, with a
        policy determining the specific type. Each backend contains a
        traits class for the specific case.

        backend<target>
        there are traits for the target.
        - type refers to the architecture specific, like the
          differences between cuda and x86.

        The execute_traits::backend_t (bad name) is responsible for
        the "inner loop nests". The
        loop<execute_traits::backend_t>::run_loop will use that to do
        whatever he has to do, for instance, the x86_backend will
        iterate over the functors of the MSS using the for_each
        available there.

        - Similarly, the definition (specialization) is contained in the

        - This contains:
        - - (INTERFACE) pointer<>::type that returns the first argument to instantiate the storage class
        - - (INTERFACE) storage_traits::storage_t to get the storage type to be used with the backend
        - - (INTERNAL) for_each that is used to invoke the different things for different stencils in the MSS
    */
    template <class BackendTarget>
    struct backend_base {

#ifdef __CUDACC__
        GT_STATIC_ASSERT((std::is_same<BackendTarget, target::cuda>::value),
            "Beware: you are compiling with nvcc, and most probably "
            "want to use the cuda backend, but the backend you are "
            "instantiating is another one!!");
#endif

        typedef storage_traits<BackendTarget> storage_traits_t;

        using backend_target_t = BackendTarget;

        /**
            Method to retrieve a global parameter
         */
        template <typename T>
        static typename storage_traits_t::template data_store_t<T,
            typename storage_traits_t::template special_storage_info_t<0, selector<0u>>>
        make_global_parameter(T const &t) {
            typename storage_traits_t::template special_storage_info_t<0, selector<0u>> si(1);
            typename storage_traits_t::template data_store_t<T, decltype(si)> ds(si);
            make_host_view(ds)(0) = t;
            ds.sync();
            return ds;
        }

        /**
            Method to update a global parameter
         */
        template <typename T, typename V>
        static void update_global_parameter(T &gp, V const &new_val) {
            gp.sync();
            auto view = make_host_view(gp);
            assert(check_consistency(gp, view) && "Cannot create a valid view to a global parameter. Properly synced?");
            view(0) = new_val;
            gp.sync();
        }
    };

} // namespace gridtools
