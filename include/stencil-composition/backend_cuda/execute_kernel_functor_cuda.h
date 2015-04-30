#pragma once
#include "../iteration_policy.h"
#include "../backend_traits_fwd.h"
#include "backend_traits_cuda.h"


template<typename T> struct printo{BOOST_MPL_ASSERT_MSG((false), OOOOOOOOOOO, (T));};
namespace gridtools {


namespace _impl_cuda {
    template <typename RunFunctorArguments,
              typename LocalDomain>
    __global__
    void do_it_on_gpu(LocalDomain const * __restrict__ l_domain, typename RunFunctorArguments::coords_t const* coords,
            const int starti, const int startj, const uint_t nx, const uint_t ny) {

        typedef typename LocalDomain::iterate_domain_t iterate_domain_t;
        typedef typename RunFunctorArguments::execution_type_t execution_type_t;

        typedef typename gridtools::backend_traits_from_id<
            RunFunctorArguments::backend_id_t::value
        >::block_size_t block_size_t;

        int i = blockIdx.x * block_size_t::i_size_t::value + threadIdx.x;
        int j = blockIdx.y * block_size_t::j_size_t::value + threadIdx.y;

        typedef typename LocalDomain::iterate_domain_t iterate_domain_t;
        __shared__
        typename iterate_domain_t::value_type* data_pointer[iterate_domain_t::N_DATA_POINTERS];

        //Doing construction and assignment before the following 'if', so that we can
        //exploit parallel shared memory initialization
        iterate_domain_t it_domain(*l_domain);
        it_domain.template assign_storage_pointers<backend_traits_from_id<enumtype::Cuda> >(
                (void**)(static_cast<typename iterate_domain_t::value_type**>(data_pointer))
        );
        __syncthreads();

//        if ((i < nx) && (j < ny)) {
        if(true){
            it_domain.assign_ij<0>(i+starti,0);
            it_domain.assign_ij<1>(j+startj,0);

            typedef typename boost::mpl::front<typename RunFunctorArguments::loop_intervals_t>::type interval;
            typedef typename index_to_level<typename interval::first>::type from;
            typedef typename index_to_level<typename interval::second>::type to;
            typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;

            //printf("setting the start to: %d \n",coords->template value_at< iteration_policy::from >() );
            //setting the initial k level (for backward/parallel iterations it is not 0)
            if( !(iteration_policy::value==enumtype::forward) )
                it_domain.set_k_start( coords->template value_at< iteration_policy::from >() );

            for_each<typename RunFunctorArguments::loop_intervals_t>
                (_impl::run_f_on_interval
                 <
                     execution_type_t, RunFunctorArguments
                  >(it_domain,*coords)
                );
        }
    }
} // namespace _impl_cuda


template <typename RunFunctorArguments >
struct execute_kernel_functor_cuda
{
    BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
    typedef typename RunFunctorArguments::local_domain_t local_domain_t;
    typedef typename RunFunctorArguments::coords_t coords_t;

    /**
       @brief core of the kernel execution
       \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
    */
    explicit execute_kernel_functor_cuda(const local_domain_t& local_domain, const coords_t& coords,
            const uint_t block_idx_i, const uint_t block_idx_j)
    : m_local_domain(local_domain)
    , m_coords(coords)
    , m_block_idx_i(block_idx_i)
    , m_block_idx_j(block_idx_j)
    {}

    void operator()()
    {
#ifndef NDEBUG
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
//        uint_t nx = (uint_t) (m_coords.i_high_bound() + range_t::iplus::value - (m_coords.i_low_bound() + range_t::iminus::value)+1);
//        uint_t ny = (uint_t) (m_coords.j_high_bound() + range_t::jplus::value - (m_coords.j_low_bound() + range_t::jminus::value)+1);
        uint_t nx = (uint_t) (m_coords.i_high_bound() - m_coords.i_low_bound() +1);
        uint_t ny = (uint_t) (m_coords.j_high_bound() - m_coords.j_low_bound() +1);

//        int ntx = 32, nty = 8, ntz = 1;

        typedef typename RunFunctorArguments::block_size_t block_size_t;

        typedef typename boost::mpl::fold<
            typename RunFunctorArguments::range_sizes_t,
            range<0,0,0,0>,
            enclosing_range<boost::mpl::_1, boost::mpl::_2>
        >::type maximum_range_t;

        typedef block_size<
            block_size_t::i_size_t::value,
            (block_size_t::j_size_t::value - maximum_range_t::jminus::value + maximum_range_t::jplus::value +
                    (maximum_range_t::iminus::value != 0 ? 1 : 0) + (maximum_range_t::iplus::value != 0 ? 1 : 0)
            )/ ((maximum_range_t::iminus::value != 0  || maximum_range_t::iplus::value != 0 ) ? 2 : 1)
        > cuda_block_size_t;
printf("PEPE %d %d %d %d %d %d \n", block_size_t::j_size_t::value, maximum_range_t::jminus::value,
        maximum_range_t::jplus::value, maximum_range_t::iminus::value, maximum_range_t::iplus::value,
        cuda_block_size_t::j_size_t::value);

        dim3 threads(cuda_block_size_t::i_size_t::value, cuda_block_size_t::j_size_t::value, 1);

        const int ntx = cuda_block_size_t::i_size_t::value;
        const int nty = cuda_block_size_t::j_size_t::value;
        const int ntz = 1;
        const int nbx = (nx + ntx - 1) / ntx;
        const int nby = (ny + nty - 1) / nty;
        const int nbz = 1;
        dim3 blocks(nbx, nby, nbz);

        typedef run_functor_arguments<
            RunFunctorArguments::backend_id_t::value,
            cuda_block_size_t,
            typename RunFunctorArguments::functor_list_t,
            typename RunFunctorArguments::esf_args_map_sequence_t,
            typename RunFunctorArguments::loop_intervals_t,
            typename RunFunctorArguments::functors_map_t,
            typename RunFunctorArguments::range_sizes_t,
            typename RunFunctorArguments::local_domain_t,
            typename RunFunctorArguments::coords_t,
            typename RunFunctorArguments::execution_type_t,
            RunFunctorArguments::s_strategy_id
        > run_functor_arguments_cuda_t;
#ifndef NDEBUG
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
        cudaDeviceSynchronize();

    }
private:
    local_domain_t m_local_domain;
    const coords_t m_coords;
    const uint_t m_block_idx_i;
    const uint_t m_block_idx_j;
};

} //namespace gridtools
