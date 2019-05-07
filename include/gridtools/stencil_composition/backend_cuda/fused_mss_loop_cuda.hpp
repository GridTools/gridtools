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
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "../grid.hpp"
#include "../iteration_policy.hpp"
#include "../local_domain.hpp"
#include "../mss_components.hpp"
#include "../mss_components_metafunctions.hpp"
#include "../run_functor_arguments.hpp"
#include "basic_token_execution_cuda.hpp"
#include "block.hpp"
#include "iterate_domain_cuda.hpp"
#include "launch_kernel.hpp"
#include "run_esf_functor_cuda.hpp"

namespace gridtools {
    namespace fused_mss_loop_cuda_impl_ {
        template <class ExecutionType>
        enable_if_t<!execute::is_parallel<ExecutionType>::value, uint_t> blocks_required_z(uint_t) {
            return 1;
        }

        template <class ExecutionType>
        enable_if_t<execute::is_parallel<ExecutionType>::value, uint_t> blocks_required_z(uint_t nz) {
            return (nz + ExecutionType::block_size - 1) / ExecutionType::block_size;
        }

        template <class ExecutionType, class From, class Grid>
        GT_FUNCTION_DEVICE enable_if_t<!execute::is_parallel<ExecutionType>::value, int_t> compute_kblock(
            Grid const &grid) {
            return grid.template value_at<From>() - grid.k_min();
        };

        template <class ExecutionType, class From, class Grid>
        GT_FUNCTION_DEVICE enable_if_t<execute::is_parallel<ExecutionType>::value, int_t> compute_kblock(
            Grid const &grid) {
            return max(blockIdx.z * ExecutionType::block_size, grid.template value_at<From>()) - grid.k_min();
        };

        template <class RunFunctorArguments, int_t BlockSizeI, int_t BlockSizeJ, class LocalDomain, class Grid>
        struct do_it_on_gpu_f {
            using execution_type_t = typename RunFunctorArguments::execution_type_t;
            using iterate_domain_arguments_t =
                iterate_domain_arguments<backend::cuda, LocalDomain, typename RunFunctorArguments::esf_sequence_t>;
            using iterate_domain_t = iterate_domain_cuda<iterate_domain_arguments_t>;

            GT_STATIC_ASSERT(std::is_trivially_copyable<LocalDomain>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(std::is_trivially_copyable<Grid>::value, GT_INTERNAL_ERROR);

            LocalDomain m_local_domain;
            Grid m_grid;

            GT_FUNCTION_DEVICE void operator()(int_t iblock, int_t jblock) const {
                // number of threads
                auto nx = m_grid.i_high_bound() - m_grid.i_low_bound() + 1;
                auto ny = m_grid.j_high_bound() - m_grid.j_low_bound() + 1;

                auto block_size_i = (blockIdx.x + 1) * BlockSizeI < nx ? BlockSizeI : nx - blockIdx.x * BlockSizeI;
                auto block_size_j = (blockIdx.y + 1) * BlockSizeJ < ny ? BlockSizeJ : ny - blockIdx.y * BlockSizeJ;

                // Doing construction of the iterate domain and assignment of pointers and strides
                iterate_domain_t it_domain(m_local_domain, block_size_i, block_size_j);

                using shared_iterate_domain_t = typename iterate_domain_t::shared_iterate_domain_t;
                __shared__ char shared_iterate_domain[sizeof(shared_iterate_domain_t)];
                it_domain.set_shared_iterate_domain_pointer(
                    reinterpret_cast<shared_iterate_domain_t *>(&shared_iterate_domain));

                using interval_t = GT_META_CALL(meta::first, typename RunFunctorArguments::loop_intervals_t);
                using from_t = GT_META_CALL(meta::first, interval_t);

                // initialize the indices.
                it_domain.initialize({m_grid.i_low_bound(), m_grid.j_low_bound(), m_grid.k_min()},
                    {blockIdx.x, blockIdx.y, blockIdx.z},
                    {iblock, jblock, compute_kblock<execution_type_t, from_t>(m_grid)});

                it_domain.set_block_pos(iblock, jblock);

                // execute the k interval functors
                run_functors_on_interval<RunFunctorArguments, run_esf_functor_cuda>(it_domain, m_grid);
            }
        };

        template <class Grid>
        struct mss_executor_f {
            Grid const &m_grid;

            template <class MssComponents, class LocalDomain>
            void operator()(MssComponents, LocalDomain const &local_domain) const {
                GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(is_mss_components<MssComponents>::value, GT_INTERNAL_ERROR);

                using execution_type_t = typename MssComponents::execution_engine_t;
                using run_functor_args_t = run_functor_arguments<backend::cuda,
                    typename MssComponents::linear_esf_t,
                    typename MssComponents::loop_intervals_t,
                    execution_type_t>;
                using max_extent_t = typename run_functor_args_t::max_extent_t;

                // number of grid points that a cuda block covers
                static constexpr int_t block_size_i = GT_DEFAULT_TILE_I;
                static constexpr int_t block_size_j = GT_DEFAULT_TILE_J;

                // number of blocks required
                uint_t xblocks = (m_grid.i_high_bound() - m_grid.i_low_bound() + block_size_i) / block_size_i;
                uint_t yblocks = (m_grid.j_high_bound() - m_grid.j_low_bound() + block_size_j) / block_size_j;
                uint_t zblocks = blocks_required_z<execution_type_t>(m_grid.k_total_length());

                launch_kernel<max_extent_t, block_size_i, block_size_j>({xblocks, yblocks, zblocks},
                    do_it_on_gpu_f<run_functor_args_t, block_size_i, block_size_j, LocalDomain, Grid>{
                        local_domain, m_grid});
            }
        };
    } // namespace fused_mss_loop_cuda_impl_
    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomains, class Grid>
    void fused_mss_loop(backend::cuda, LocalDomains const &local_domains, const Grid &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using namespace std::placeholders;
        tuple_util::for_each(fused_mss_loop_cuda_impl_::mss_executor_f<Grid>{grid}, MssComponents{}, local_domains);
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    std::true_type mss_fuse_esfs(backend::cuda);
} // namespace gridtools
