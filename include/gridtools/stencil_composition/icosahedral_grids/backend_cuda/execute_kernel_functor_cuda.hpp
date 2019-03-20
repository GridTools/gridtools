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

#ifdef GT_VERBOSE
#include <iostream>
#endif

#include "../../../common/cuda_util.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/gt_assert.hpp"
#include "../../backend_cuda/basic_token_execution_cuda.hpp"
#include "../../backend_cuda/execute_kernel_functor_cuda_common.hpp"
#include "../../backend_cuda/run_esf_functor_cuda.hpp"
#include "../../backend_traits_fwd.hpp"
#include "../../block.hpp"
#include "../../iteration_policy.hpp"
#include "./iterate_domain_cuda.hpp"

namespace gridtools {

    namespace _impl_iccuda {

        template <int VBoundary>
        struct padded_boundary
            : boost::mpl::integral_c<int, VBoundary <= 1 ? 1 : (VBoundary <= 2 ? 2 : (VBoundary <= 4 ? 4 : 8))> {
            GT_STATIC_ASSERT(VBoundary >= 0 && VBoundary <= 8, GT_INTERNAL_ERROR);
        };

        template <typename RunFunctorArguments, size_t NumThreads>
        __global__ void __launch_bounds__(NumThreads)
            do_it_on_gpu(typename RunFunctorArguments::local_domain_t const l_domain,
                typename RunFunctorArguments::grid_t const grid) {

            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            typedef typename RunFunctorArguments::max_extent_t max_extent_t;

            using iterate_domain_arguments_t = iterate_domain_arguments<typename RunFunctorArguments::backend_ids_t,
                typename RunFunctorArguments::local_domain_t,
                typename RunFunctorArguments::esf_sequence_t,
                typename RunFunctorArguments::cache_sequence_t,
                typename RunFunctorArguments::grid_t>;

            using iterate_domain_t = iterate_domain_cuda<iterate_domain_arguments_t>;

            typedef backend_traits_from_id<target::cuda> backend_traits_t;

            const uint_t nx = (uint_t)(grid.i_high_bound() - grid.i_low_bound() + 1);
            const uint_t ny = (uint_t)(grid.j_high_bound() - grid.j_low_bound() + 1);

            static constexpr auto backend = typename RunFunctorArguments::backend_ids_t{};
            static constexpr uint_t ntx = block_i_size(backend);
            static constexpr uint_t nty = block_j_size(backend);

            const uint_t block_size_i = (blockIdx.x + 1) * ntx < nx ? ntx : nx - blockIdx.x * ntx;
            const uint_t block_size_j = (blockIdx.y + 1) * nty < ny ? nty : ny - blockIdx.y * nty;

            __shared__ typename iterate_domain_t::shared_iterate_domain_t shared_iterate_domain;

            // Doing construction of the iterate domain and assignment of pointers and strides
            iterate_domain_t it_domain(l_domain, block_size_i, block_size_j);

            it_domain.set_shared_iterate_domain_pointer(&shared_iterate_domain);

            it_domain.assign_stride_pointers();

            __syncthreads();

            // computing the global position in the physical domain
            /*
             *  In a typical cuda block we have the following regions
             *
             *    aa bbbbbbbb cc
             *    aa bbbbbbbb cc
             *
             *    hh dddddddd ii
             *    hh dddddddd ii
             *    hh dddddddd ii
             *    hh dddddddd ii
             *
             *    ee ffffffff gg
             *    ee ffffffff gg
             *
             * Regions b,d,f have warp (or multiple of warp size)
             * Size of regions a, c, h, i, e, g are determined by max_extent_t
             * Regions b,d,f are easily executed by dedicated warps (one warp for each line).
             * Regions (a,h,e) and (c,i,g) are executed by two specialized warp
             *
             */
            // jboundary_limit determines the number of warps required to execute (b,d,f)
            // TODO FUSING
            static constexpr auto jboundary_limit = int(nty) + max_extent_t::jplus::value - max_extent_t::jminus::value;
            // iminus_limit adds to jboundary_limit an additional warp for regions (a,h,e)
            static constexpr auto iminus_limit = jboundary_limit + (max_extent_t::iminus::value < 0 ? 1 : 0);
            // iminus_limit adds to iminus_limit an additional warp for regions (c,i,g)
            static constexpr auto iplus_limit = iminus_limit + (max_extent_t::iplus::value > 0 ? 1 : 0);

            // The kernel allocate enough warps to execute all halos of all ESFs.
            // The max_extent_t is the enclosing extent of all the ESFs
            // (i,j) is the position (in the global domain, minus initial halos which are accounted with istart, jstart
            // args)
            // of this thread within the physical block
            // (iblock, jblock) are relative positions of the thread within the block. Grid positions in the halos of
            // the block
            //   get negative values

            int iblock = max_extent_t::iminus::value - 1;
            int jblock = max_extent_t::jminus::value - 1;
            if (threadIdx.y < jboundary_limit) {
                iblock = threadIdx.x;
                jblock = (int)threadIdx.y + max_extent_t::jminus::value;
            } else if (threadIdx.y < iminus_limit) {
                static constexpr auto padded_boundary_ = padded_boundary<-max_extent_t::iminus::value>::value;
                // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough threads
                GT_STATIC_ASSERT(
                    jboundary_limit * padded_boundary_ <= block_i_size(backend_ids<target::cuda, strategy::block>{}),
                    GT_INTERNAL_ERROR);
                iblock = -padded_boundary_ + (int)threadIdx.x % padded_boundary_;
                jblock = (int)threadIdx.x / padded_boundary_ + max_extent_t::jminus::value;
            } else if (threadIdx.y < iplus_limit) {
                const int padded_boundary_ = padded_boundary<max_extent_t::iplus::value>::value;
                // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough threads
                GT_STATIC_ASSERT(
                    jboundary_limit * padded_boundary_ <= block_i_size(backend_ids<target::cuda, strategy::block>{}),
                    GT_INTERNAL_ERROR);

                iblock = threadIdx.x % padded_boundary_ + ntx;
                jblock = (int)threadIdx.x / padded_boundary_ + max_extent_t::jminus::value;
            }

            using interval_t = GT_META_CALL(meta::first, typename RunFunctorArguments::loop_intervals_t);
            using from_t = GT_META_CALL(meta::first, interval_t);

            const int_t kblock = impl_::compute_kblock<execution_type_t>::template get<from_t>(grid);
            it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
                {blockIdx.x, blockIdx.y, blockIdx.z},
                {iblock, jblock, kblock});
            it_domain.set_block_pos(iblock, jblock);

            // execute the k interval functors
            run_functors_on_interval<RunFunctorArguments, run_esf_functor_cuda>(it_domain, grid);
        }

    } // namespace _impl_iccuda

    namespace icgrid {

        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_cuda {
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;

            GT_STATIC_ASSERT(cuda_util::is_cloneable<local_domain_t>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(cuda_util::is_cloneable<grid_t>::value, GT_INTERNAL_ERROR);

            // ctor
            explicit execute_kernel_functor_cuda(const local_domain_t &local_domain, const grid_t &grid)
                : m_local_domain(local_domain), m_grid(grid) {}

            void operator()() {
#ifdef GT_VERBOSE
                short_t count;
                GT_CUDA_CHECK(cudaGetDeviceCount(&count));

                if (count) {
                    cudaDeviceProp prop;
                    GT_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
                    std::cout << "total global memory " << prop.totalGlobalMem << std::endl;
                    std::cout << "shared memory per block " << prop.sharedMemPerBlock << std::endl;
                    std::cout << "registers per block " << prop.regsPerBlock << std::endl;
                    std::cout << "maximum threads per block " << prop.maxThreadsPerBlock << std::endl;
                    std::cout << "maximum threads dimension " << prop.maxThreadsDim << std::endl;
                    std::cout << "clock rate " << prop.clockRate << std::endl;
                    std::cout << "total const memory " << prop.totalConstMem << std::endl;
                    std::cout << "compute capability " << prop.major << "." << prop.minor << std::endl;
                    std::cout << "multiprocessors count " << prop.multiProcessorCount << std::endl;
                    std::cout << "CUDA compute mode (0=default, 1=exclusive, 2=prohibited, 3=exclusive process) "
                              << prop.computeMode << std::endl;
                    std::cout << "concurrent kernels " << prop.concurrentKernels << std::endl;
                    std::cout << "Number of asynchronous engines  " << prop.asyncEngineCount << std::endl;
                    std::cout << "unified addressing " << prop.unifiedAddressing << std::endl;
                    std::cout << "memoryClockRate " << prop.memoryClockRate << std::endl;
                    std::cout << "memoryBusWidth " << prop.memoryBusWidth << std::endl;
                    std::cout << "l2CacheSize " << prop.l2CacheSize << std::endl;
                    std::cout << "maxThreadsPerMultiProcessor " << prop.maxThreadsPerMultiProcessor << std::endl;
                }
#endif

                // number of threads
                const uint_t nx = (uint_t)(m_grid.i_high_bound() - m_grid.i_low_bound() + 1);
                const uint_t ny = (uint_t)(m_grid.j_high_bound() - m_grid.j_low_bound() + 1);
                const uint_t nz = m_grid.k_total_length();

                static constexpr auto backend = typename RunFunctorArguments::backend_ids_t{};

                // number of grid points that a cuda block covers
                static constexpr uint_t ntx = block_i_size(backend);
                static constexpr uint_t nty = block_j_size(backend);
                static constexpr uint_t ntz = 1;

                using max_extent_t = typename RunFunctorArguments::max_extent_t;
                static constexpr uint_t halo_processing_warps =
                    max_extent_t::jplus::value - max_extent_t::jminus::value +
                    (max_extent_t::iminus::value < 0 ? 1 : 0) + (max_extent_t::iplus::value > 0 ? 1 : 0);

                dim3 threads(ntx, nty + halo_processing_warps, ntz);

                // number of blocks required
                const uint_t nbx = (nx + ntx - 1) / ntx;
                const uint_t nby = (ny + nty - 1) / nty;
                using execution_type_t = typename RunFunctorArguments::execution_type_t;
                const uint_t nbz = impl_::blocks_required_z<execution_type_t>::get(nz);

                dim3 blocks(nbx, nby, nbz);

#ifdef GT_VERBOSE
                printf("ntx = %d, nty = %d, ntz = %d\n", ntx, nty, ntz);
                printf("nbx = %d, nby = %d, nbz = %d\n", nbx, nby, nbz);
                printf("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);
#endif

                _impl_iccuda::do_it_on_gpu<RunFunctorArguments, ntx *(nty + halo_processing_warps)>
                    <<<blocks, threads>>>(m_local_domain, m_grid);
#ifndef NDEBUG
                GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
                GT_CUDA_CHECK(cudaGetLastError());
#endif
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
        };

    } // namespace icgrid

} // namespace gridtools
