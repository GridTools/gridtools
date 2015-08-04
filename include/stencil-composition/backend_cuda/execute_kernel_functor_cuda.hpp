#pragma once
#include "../iteration_policy.hpp"
#include "../backend_traits_fwd.hpp"
#include "backend_traits_cuda.hpp"
#include "../iterate_domain_aux.hpp"
#include "shared_iterate_domain.hpp"
#include "common/gt_assert.hpp"

namespace gridtools {


namespace _impl_cuda {
    template <typename RunFunctorArguments,
              typename LocalDomain>
    __global__
    void do_it_on_gpu(LocalDomain const * RESTRICT l_domain, typename RunFunctorArguments::coords_t const* coords,
            const int starti, const int startj, const uint_t nx, const uint_t ny) {

        assert(l_domain);
        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
        typedef typename RunFunctorArguments::execution_type_t execution_type_t;

        typedef typename RunFunctorArguments::physical_domain_block_size_t block_size_t;

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
        typedef backend_traits_from_id<enumtype::Cuda> backend_traits_t;
        typedef strides_cached<iterate_domain_t::N_STORAGES-1, typename LocalDomain::esf_args> strides_t;
        typedef array<void* RESTRICT,iterate_domain_t::N_DATA_POINTERS> data_pointer_t;
        typedef shared_iterate_domain<
            data_pointer_t,
            strides_t,
            typename iterate_domain_t::iterate_domain_cache_t::ij_caches_tuple_t
        > shared_iterate_domain_t;

        const uint_t block_size_i = (blockIdx.x+1) * block_size_t::i_size_t::value < nx ?
                block_size_t::i_size_t::value : nx - blockIdx.x * block_size_t::i_size_t::value ;
        const uint_t block_size_j = (blockIdx.y+1) * block_size_t::j_size_t::value < ny ?
                block_size_t::j_size_t::value : ny - blockIdx.y * block_size_t::j_size_t::value ;

        __shared__ shared_iterate_domain_t shared_iterate_domain;

        //Doing construction of the ierate domain and assignment of pointers and strides
        iterate_domain_t it_domain(*l_domain, block_size_i, block_size_j);

        it_domain.set_shared_iterate_domain_pointer_impl(&shared_iterate_domain);

        it_domain.template assign_storage_pointers<backend_traits_t >();
        it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();

        __syncthreads();

        //computing the global position in the physical domain
        const int i = blockIdx.x * block_size_t::i_size_t::value + threadIdx.x;
        const int j = blockIdx.y * block_size_t::j_size_t::value + threadIdx.y;

        it_domain.set_index(0);

        //initialize the indices
        it_domain.template initialize<0>(i+starti, blockIdx.x);
        it_domain.template initialize<1>(j+startj, blockIdx.y);

        typedef typename boost::mpl::front<typename RunFunctorArguments::loop_intervals_t>::type interval;
        typedef typename index_to_level<typename interval::first>::type from;
        typedef typename index_to_level<typename interval::second>::type to;
        typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;

        //setting the initial k level (for backward/parallel iterations it is not 0)
        if( !(iteration_policy::value==enumtype::forward) )
            it_domain.initialize<2>( coords->template value_at< iteration_policy::from >() );

        //execute the k interval functors
        for_each<typename RunFunctorArguments::loop_intervals_t>
            (_impl::run_f_on_interval<execution_type_t, RunFunctorArguments>(it_domain,*coords) );
    }
} // namespace _impl_cuda


/**
 * @brief main functor that setups the CUDA kernel for a MSS and launchs it
 * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
 */
template <typename RunFunctorArguments >
struct execute_kernel_functor_cuda
{
    GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
    typedef typename RunFunctorArguments::local_domain_t local_domain_t;
    typedef typename RunFunctorArguments::coords_t coords_t;

    //ctor
    explicit execute_kernel_functor_cuda(const local_domain_t& local_domain, const coords_t& coords,
            const uint_t block_idx_i, const uint_t block_idx_j)
    : m_local_domain(local_domain)
    , m_coords(coords)
    , m_block_idx_i(block_idx_i)
    , m_block_idx_j(block_idx_j)
    {}

    void operator()()
    {
#ifdef __VERBOSE__
        short_t count;
        cudaGetDeviceCount ( &count  );

        if(count)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "total global memory "<<       prop.totalGlobalMem<<std::endl;
            std::cout << "shared memory per block "<<   prop.sharedMemPerBlock<<std::endl;
            std::cout << "registers per block "<<       prop.regsPerBlock<<std::endl;
            std::cout << "maximum threads per block "<< prop.maxThreadsPerBlock <<std::endl;
            std::cout << "maximum threads dimension "<< prop.maxThreadsDim <<std::endl;
            std::cout << "clock rate "<<                prop.clockRate <<std::endl;
            std::cout << "total const memory "<<        prop.totalConstMem <<std::endl;
            std::cout << "compute capability "<<        prop.major<<"."<<prop.minor <<std::endl;
            std::cout << "multiprocessors count "<< prop.multiProcessorCount <<std::endl;
            std::cout << "CUDA compute mode (0=default, 1=exclusive, 2=prohibited, 3=exclusive process) "<< prop.computeMode <<std::endl;
            std::cout << "concurrent kernels "<< prop.concurrentKernels <<std::endl;
            std::cout << "Number of asynchronous engines  "<< prop.asyncEngineCount <<std::endl;
            std::cout << "unified addressing "<< prop.unifiedAddressing <<std::endl;
            std::cout << "memoryClockRate "<< prop.memoryClockRate <<std::endl;
            std::cout << "memoryBusWidth "<< prop.memoryBusWidth <<std::endl;
            std::cout << "l2CacheSize "<< prop.l2CacheSize <<std::endl;
            std::cout << "maxThreadsPerMultiProcessor "<< prop.maxThreadsPerMultiProcessor <<std::endl;
        }
#endif
        m_local_domain.clone_to_gpu();
        m_coords.clone_to_gpu();

        local_domain_t *local_domain_gp = m_local_domain.gpu_object_ptr;

        coords_t const *coords_gp = m_coords.gpu_object_ptr;

        // number of threads
        const uint_t nx = (uint_t) (m_coords.i_high_bound() - m_coords.i_low_bound() +1);
        const uint_t ny = (uint_t) (m_coords.j_high_bound() - m_coords.j_low_bound() +1);

        typedef typename RunFunctorArguments::physical_domain_block_size_t block_size_t;

        //compute the union (or enclosing) range for the ranges of all ESFs.
        //This maximum range of all ESF will determine the size of the CUDA block:
        // *  If there are redundant computations to be executed at the IMinus or IPlus halos,
        //    each CUDA thread will execute two grid points (one at the core of the block and
        //    another within one of the halo regions)
        // *  Otherwise each CUDA thread executes only one grid point.
        // Based on the previous we compute the size of the CUDA block required.
        typedef typename boost::mpl::fold<
            typename RunFunctorArguments::range_sizes_t,
            range<0,0,0,0,0,0>,
            enclosing_range<boost::mpl::_1, boost::mpl::_2>
        >::type maximum_range_t;

        typedef block_size<
            block_size_t::i_size_t::value,
            (block_size_t::j_size_t::value - maximum_range_t::jminus::value + maximum_range_t::jplus::value +
                    (maximum_range_t::iminus::value != 0 ? 1 : 0) + (maximum_range_t::iplus::value != 0 ? 1 : 0)
            )/ ((maximum_range_t::iminus::value != 0  || maximum_range_t::iplus::value != 0 ) ? 2 : 1)
        > cuda_block_size_t;

        //number of grid points that a cuda block covers
        const uint_t ntx = block_size_t::i_size_t::value;
        const uint_t nty = block_size_t::j_size_t::value;
        const uint_t ntz = 1;
        dim3 threads(cuda_block_size_t::i_size_t::value, cuda_block_size_t::j_size_t::value, ntz);

        //number of blocks required
        const uint_t nbx = (nx + ntx - 1) / ntx;
        const uint_t nby = (ny + nty - 1) / nty;
        const uint_t nbz = 1;

        dim3 blocks(nbx, nby, nbz);

        //re-create the run functor arguments, replacing the processing elements block size
        // with the corresponding, recently computed, block size
        typedef run_functor_arguments<
            RunFunctorArguments::backend_id_t::value,
            cuda_block_size_t,
            typename RunFunctorArguments::physical_domain_block_size_t,
            typename RunFunctorArguments::functor_list_t,
            typename RunFunctorArguments::esf_sequence_t,
            typename RunFunctorArguments::esf_args_map_sequence_t,
            typename RunFunctorArguments::loop_intervals_t,
            typename RunFunctorArguments::functors_map_t,
            typename RunFunctorArguments::range_sizes_t,
            typename RunFunctorArguments::local_domain_t,
            typename RunFunctorArguments::cache_sequence_t,
            typename RunFunctorArguments::coords_t,
            typename RunFunctorArguments::execution_type_t,
            RunFunctorArguments::s_strategy_id
        > run_functor_arguments_cuda_t;

#ifdef __VERBOSE__
            printf("ntx = %d, nty = %d, ntz = %d\n",ntx, nty, ntz);
            printf("nbx = %d, nby = %d, nbz = %d\n",nbx, nby, nbz);
            printf("nx = %d, ny = %d, nz = 1\n",nx, ny);
#endif

        _impl_cuda::do_it_on_gpu<run_functor_arguments_cuda_t, local_domain_t><<<blocks, threads>>>//<<<nbx*nby, ntx*nty>>>
            (local_domain_gp, coords_gp,
                 m_coords.i_low_bound(),
                 m_coords.j_low_bound(),
                 (nx),
                 (ny)
            );

        //TODOCOSUNA we do not need this. It will block the host, and we want to continue doing other stuff
        cudaDeviceSynchronize();

    }
private:
    const local_domain_t& m_local_domain;
    const coords_t& m_coords;
    const uint_t m_block_idx_i;
    const uint_t m_block_idx_j;
};

} //namespace gridtools
