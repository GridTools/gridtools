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

#include "../../../common/cuda_util.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/meta.hpp"
#include "../../../common/gt_assert.hpp"
#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../backend_traits_fwd.hpp"
#include "../../iteration_policy.hpp"
#include "./iterate_domain_cuda.hpp"

namespace gridtools {

    namespace _impl_iccuda {

        template <int VBoundary>
        struct padded_boundary
            : boost::mpl::integral_c<int, VBoundary <= 1 ? 1 : (VBoundary <= 2 ? 2 : (VBoundary <= 4 ? 4 : 8))> {
            GRIDTOOLS_STATIC_ASSERT(VBoundary >= 0 && VBoundary <= 8, GT_INTERNAL_ERROR);
        };

        template <typename RunFunctorArguments, size_t NumThreads>
        __global__ void __launch_bounds__(NumThreads)
            do_it_on_gpu(typename RunFunctorArguments::local_domain_t const l_domain,
                typename RunFunctorArguments::grid_t const grid) {

            typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            typedef typename RunFunctorArguments::physical_domain_block_size_t block_size_t;
            typedef typename RunFunctorArguments::extent_sizes_t extent_sizes_t;

            typedef typename RunFunctorArguments::max_extent_t max_extent_t;
            typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
            typedef typename RunFunctorArguments::async_esf_map_t async_esf_map_t;

            typedef backend_traits_from_id<enumtype::Cuda> backend_traits_t;
            typedef typename iterate_domain_t::strides_cached_t strides_t;
            typedef typename iterate_domain_t::data_ptr_cached_t data_ptr_cached_t;
            typedef shared_iterate_domain<data_ptr_cached_t,
                strides_t,
                max_extent_t,
                typename iterate_domain_t::iterate_domain_cache_t::ij_caches_tuple_t>
                shared_iterate_domain_t;

            const uint_t nx = (uint_t)(grid.i_high_bound() - grid.i_low_bound() + 1);
            const uint_t ny = (uint_t)(grid.j_high_bound() - grid.j_low_bound() + 1);

            const uint_t block_size_i = (blockIdx.x + 1) * block_size_t::i_size_t::value < nx
                                            ? block_size_t::i_size_t::value
                                            : nx - blockIdx.x * block_size_t::i_size_t::value;
            const uint_t block_size_j = (blockIdx.y + 1) * block_size_t::j_size_t::value < ny
                                            ? block_size_t::j_size_t::value
                                            : ny - blockIdx.y * block_size_t::j_size_t::value;

            __shared__ shared_iterate_domain_t shared_iterate_domain;

            // Doing construction of the iterate domain and assignment of pointers and strides
            iterate_domain_t it_domain(l_domain, grid.grid_topology(), block_size_i, block_size_j);

            it_domain.set_shared_iterate_domain_pointer_impl(&shared_iterate_domain);

            it_domain.template assign_storage_pointers<backend_traits_t>();
            it_domain.template assign_stride_pointers<backend_traits_t, strides_t>();

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
            const int jboundary_limit =
                block_size_t::j_size_t::value - max_extent_t::jminus::value + max_extent_t::jplus::value;
            // iminus_limit adds to jboundary_limit an additional warp for regions (a,h,e)
            const int iminus_limit = jboundary_limit + (max_extent_t::iminus::value < 0 ? 1 : 0);
            // iminus_limit adds to iminus_limit an additional warp for regions (c,i,g)
            const int iplus_limit = iminus_limit + (max_extent_t::iplus::value > 0 ? 1 : 0);

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
                const int padded_boundary_ = padded_boundary<-max_extent_t::iminus::value>::value;
                // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough threads
                assert((block_size_t::j_size_t::value - max_extent_t::jminus::value + max_extent_t::jplus::value) *
                           padded_boundary_ <=
                       enumtype::vector_width);

                iblock = -padded_boundary_ + (int)threadIdx.x % padded_boundary_;
                jblock = (int)threadIdx.x / padded_boundary_ + max_extent_t::jminus::value;
            } else if (threadIdx.y < iplus_limit) {
                const int padded_boundary_ = padded_boundary<max_extent_t::iplus::value>::value;
                // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough threads
                assert((block_size_t::j_size_t::value - max_extent_t::jminus::value + max_extent_t::jplus::value) *
                           padded_boundary_ <=
                       enumtype::vector_width);

                iblock = threadIdx.x % padded_boundary_ + block_size_t::i_size_t::value;
                jblock = (int)threadIdx.x / padded_boundary_ + max_extent_t::jminus::value;
            }

            typedef typename boost::mpl::front<typename RunFunctorArguments::loop_intervals_t>::type interval;
            typedef typename index_to_level<typename interval::first>::type from;
            typedef typename index_to_level<typename interval::second>::type to;
            typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy_t;

            it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
                {blockIdx.x, blockIdx.y, 0},
                {iblock,
                    jblock,
                    static_cast<int_t>(grid.template value_at<iteration_policy_t::from>() - grid.k_min())});
            it_domain.set_block_pos(iblock, jblock);

            // execute the k interval functors
            boost::mpl::for_each<typename RunFunctorArguments::loop_intervals_t>(
                _impl::run_f_on_interval<execution_type_t, RunFunctorArguments>(it_domain, grid));
        }

    } // namespace _impl_iccuda

    namespace icgrid {

        /**
         * @brief main functor that setups the CUDA kernel for a MSS and launchs it
         * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
         */
        template <typename RunFunctorArguments>
        struct execute_kernel_functor_cuda {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;

            GRIDTOOLS_STATIC_ASSERT(cuda_util::is_cloneable<local_domain_t>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(cuda_util::is_cloneable<grid_t>::value, GT_INTERNAL_ERROR);

            // ctor
            explicit execute_kernel_functor_cuda(const local_domain_t &local_domain, const grid_t &grid)
                : m_local_domain(local_domain), m_grid(grid) {}

            void operator()() {
#ifdef VERBOSE
                short_t count;
                cudaGetDeviceCount(&count);

                if (count) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, 0);
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

                typedef typename RunFunctorArguments::physical_domain_block_size_t block_size_t;

                typedef typename RunFunctorArguments::cuda_block_size_t cuda_block_size_t;

                // number of grid points that a cuda block covers
                constexpr uint_t ntx = block_size_t::i_size_t::value;
                constexpr uint_t nty = block_size_t::j_size_t::value;
                constexpr uint_t ntz = 1;
                dim3 threads(cuda_block_size_t::i_size_t::value, cuda_block_size_t::j_size_t::value, ntz);

                // number of blocks required
                const uint_t nbx = (nx + ntx - 1) / ntx;
                const uint_t nby = (ny + nty - 1) / nty;
                const uint_t nbz = 1;

                dim3 blocks(nbx, nby, nbz);

                // re-create the run functor arguments, replacing the processing elements block size
                // with the corresponding, recently computed, block size
                using run_functor_arguments_cuda_t =
                    GT_META_CALL(meta::replace_at_c, (RunFunctorArguments, 1, cuda_block_size_t));

#ifdef VERBOSE
                printf("ntx = %d, nty = %d, ntz = %d\n", ntx, nty, ntz);
                printf("nbx = %d, nby = %d, nbz = %d\n", nbx, nby, nbz);
                printf("nx = %d, ny = %d, nz = 1\n", nx, ny);
#endif

                constexpr size_t num_threads = cuda_block_size_t::i_size_t::value * cuda_block_size_t::j_size_t::value;
                _impl_iccuda::do_it_on_gpu<run_functor_arguments_cuda_t, num_threads>
                    <<<blocks, threads>>>(m_local_domain, m_grid);
#ifndef NDEBUG
                cudaDeviceSynchronize();
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorName(error), __FILE__, __LINE__);
                    exit(-1);
                }
#endif
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
        };

    } // namespace icgrid

} // namespace gridtools
