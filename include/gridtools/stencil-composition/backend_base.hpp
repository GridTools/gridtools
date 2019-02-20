/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cassert>
#include <type_traits>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/selector.hpp"
#include "../storage/storage-facility.hpp"
#include "./backend_ids.hpp"
#include "./grid.hpp"
#include "./local_domain.hpp"
#include "./mss_components.hpp"

#ifdef __CUDACC__
#include "./backend_cuda/backend_traits_cuda.hpp"
#endif
#include "./backend_mc/backend_traits_mc.hpp"
#include "./backend_x86/backend_traits_x86.hpp"

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

        backend<type, strategy>
        there are traits: one for type and one for strategy.
        - type refers to the architecture specific, like the
          differences between cuda and x86.

        The backend has a member function "run" that is called by the
        "intermediate".
        The "run" method calls strategy_from_id<strategy>::loop

        - the strategy_from_id is in the specific backend_? folder, such as
        - in backend_?/backend_traits.h

        - strategy_from_id contains the tile size information and the
        - "struct loop" which has the "run_loop" member function.

        Before calling the loop::run_loop method, the backend queries
        "execute_traits" that are contained in the
        "backend_traits_t". the latter is obtained by

        backend_strategy_from_id<type>

        The execute_traits::backend_t (bad name) is responsible for
        the "inner loop nests". The
        loop<execute_traits::backend_t>::run_loop will use that to do
        whatever he has to do, for instance, the x86_backend will
        iterate over the functors of the MSS using the for_each
        available there.

        - Similarly, the definition (specialization) is contained in the
        - specific subfoled (right now in backend_?/backend_traits_?.h ).

        - This contains:
        - - (INTERFACE) pointer<>::type that returns the first argument to instantiate the storage class
        - - (INTERFACE) storage_traits::storage_t to get the storage type to be used with the backend
        - - (INTERFACE) execute_traits ?????? this was needed when backend_traits was forcely shared between x86 and
       cuda backends. Now they are separated and this may be simplified.
        - - (INTERNAL) for_each that is used to invoke the different things for different stencils in the MSS
        - - (INTERNAL) once_per_block
    */
    template <class BackendId, class GridId, class StrategyId>
    struct backend_base {

#ifdef __CUDACC__
        GT_STATIC_ASSERT((std::is_same<BackendId, target::cuda>::value),
            "Beware: you are compiling with nvcc, and most probably "
            "want to use the cuda backend, but the backend you are "
            "instantiating is another one!!");
#endif

        typedef backend_base<BackendId, GridId, StrategyId> this_type;

        typedef backend_ids<BackendId, GridId, StrategyId> backend_ids_t;

        typedef backend_traits_from_id<BackendId> backend_traits_t;
        typedef storage_traits<BackendId> storage_traits_t;
        typedef typename backend_traits_t::template select_strategy<backend_ids_t>::type strategy_traits_t;

        using strategy_id_t = StrategyId;
        using backend_id_t = BackendId;
        using grid_id_t = GridId;

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

        using make_view_f = typename backend_traits_t::make_view_f;

        using mss_fuse_esfs_strategy = typename backend_traits_t::mss_fuse_esfs_strategy;

        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * @tparam MssArray  meta array of mss
         * \tparam Grid Coordinate class with domain sizes and splitter grid
         * \tparam LocalDomains sequence of local domains
         */
        template <typename MssComponents, typename Grid, typename LocalDomains>
        static void run(Grid const &grid, LocalDomains const &local_domains) {
            // TODO: I would swap the arguments coords and local_domains, for consistency
            GT_STATIC_ASSERT((is_sequence_of<LocalDomains, is_local_domain>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

            strategy_traits_t::template fused_mss_loop<MssComponents, backend_ids_t>::run(local_domains, grid);
        }
    };

} // namespace gridtools
