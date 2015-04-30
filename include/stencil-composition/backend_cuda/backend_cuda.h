#pragma once

#include "../execution_policy.h"
#include "../heap_allocated_temps.h"
#include "../run_kernel.h"
#include "backend_traits.h"
#include "backend_traits_cuda.h"

/**
 * @file
 * \brief implements the stencil operations for a GPU backend
 */

namespace gridtools {

    /** Kernel function called from the GPU */
//    namespace _impl_cuda {
//        template <typename Arguments,
//                  typename EsfArguments,
//                  typename LocalDomain>
//        __global__
//        void do_it_on_gpu(LocalDomain const * __restrict__ l_domain, typename Arguments::coords_t const* coords, uint_t const starti, uint_t const startj, uint_t const nx, uint_t const ny) {
////             uint_t j = (blockIdx.x * blockDim.x + threadIdx.x)%ny;
////             uint_t i = (blockIdx.x * blockDim.x + threadIdx.x - j)/ny;
//            int i = blockIdx.x * blockDim.x + threadIdx.x;
//            int j = blockIdx.y * blockDim.y + threadIdx.y;
//            uint_t z = coords->template value_at<typename EsfArguments::first_hit_t>();
//
//            typedef typename EsfArguments::iterate_domain_t iterate_domain_t;
//            __shared__
//                typename iterate_domain_t::value_type* data_pointer[iterate_domain_t::N_DATA_POINTERS];
//
//            //Doing construction and assignment before the following 'if', so that we can
//            //exploit parallel shared memory initialization
//            iterate_domain_t it_domain(*l_domain);
//            it_domain.template assign_storage_pointers<backend_traits_from_id<enumtype::Cuda> >(
//                    (void**)(static_cast<typename iterate_domain_t::value_type**>(data_pointer))
//            );
//            __syncthreads();
//
//            if ((i < nx) && (j < ny)) {
//
//                it_domain.assign_ij<0>(i+starti,0);
//                it_domain.assign_ij<1>(j+startj,0);
//
//                typedef typename boost::mpl::front<typename Arguments::loop_intervals_t>::type interval;
//                typedef typename index_to_level<typename interval::first>::type from;
//                typedef typename index_to_level<typename interval::second>::type to;
//                typedef typename Arguments::execution_type_t execution_type_t;
//                typedef _impl::iteration_policy<from, to, Arguments::execution_type_t::type::iteration> iteration_policy;
//
//                //printf("setting the start to: %d \n",coords->template value_at< iteration_policy::from >() );
//                //setting the initial k level (for backward/parallel iterations it is not 0)
//                if( !(iteration_policy::value==enumtype::forward) )
//                    it_domain.set_k_start( coords->template value_at< iteration_policy::from >() );
//
//                for_each<typename Arguments::loop_intervals_t>
//                    (_impl::run_f_on_interval
//                     <
//                         execution_type_t, EsfArguments, Arguments
//                      >(it_domain,*coords)
//                    );
//            }
//
//        }

//        /**
//         * \brief this struct is the core of the ESF functor
//         */
//        template < typename Arguments >
//        struct run_functor_cuda : public _impl::run_functor < run_functor_cuda< Arguments > >
//        {
//            typedef _impl::run_functor < run_functor_cuda< Arguments > > super;
//            explicit run_functor_cuda(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
//                : super( domain_list, coords)
//                {}
//
//            explicit run_functor_cuda(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj)
//                : super(domain_list, coords, i, j, bi, bj)
//                {}
//
//        };
//
//    }//namespace _impl_cuda

    namespace _impl {
        template<typename Arguments>
        struct run_functor_backend_id<_impl_cuda::run_functor_cuda<Arguments> > : boost::mpl::integral_c<enumtype::backend, enumtype::Cuda> {};
    } //  namespace _impl
//    {
/** Partial specialization: naive implementation for the Cuda backend (2 policies specify strategy and backend)*/
    template < typename Arguments >
    struct execute_kernel_functor < _impl_cuda::run_functor_cuda<Arguments> >
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<Arguments>::value));
        typedef _impl_cuda::run_functor_cuda<Arguments> backend_t;

/**
   @brief core of the kernel execution
   \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
*/
        template <
            typename EsfArguments,
            typename LocalDomain >
        static void execute_kernel( LocalDomain& local_domain, const backend_t * f )
        {
            typedef typename Arguments::coords_t coords_type;
            // typedef typename Arguments::loop_intervals_t loop_intervals_t;
            typedef typename EsfArguments::range_t range_t;
            typedef typename EsfArguments::functor_t functor_type;
            typedef typename EsfArguments::interval_map_t interval_map_type;
            typedef typename LocalDomain::iterate_domain_t iterate_domain_t;
            typedef typename EsfArguments::first_hit_t first_hit_t;
            typedef typename Arguments::execution_type_t execution_type_t;


            /* struct extra_arguments{ */
            /*     typedef functor_type functor_t; */
            /*     typedef interval_map_type interval_map_t; */
            /*     typedef iterate_domain_t local_domain_t; */
            /*     typedef coords_type coords_t;}; */

#ifndef NDEBUG
            std::cout << "Functor " <<  functor_type() << "\n";
            std::cout << "I loop " << f->m_start[0]  + range_t::iminus::value << " -> "
                                    << f->m_start[0] + f->m_BI + range_t::iplus::value << "\n";
            std::cout << "J loop " << f->m_start[1] + range_t::jminus::value << " -> "
                                    << f->m_start[1] + f->m_BJ + range_t::jplus::value << "\n";
            std::cout <<  " ******************** " << typename EsfArguments::first_hit_t() << "\n";
            std::cout << " ******************** " << f->m_coords.template value_at<typename EsfArguments::first_hit_t>() << "\n";

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
            local_domain.clone_to_gpu();
            f->m_coords.clone_to_gpu();

            LocalDomain *local_domain_gp = local_domain.gpu_object_ptr;

//         int nx = func_->m_coords.i_high_bound() + range_t::iplus::value - (func_->m_coords.i_low_bound() + range_t::iminus::value);
//         int ny = func_->m_coords.j_high_bound() + range_t::jplus::value - (func_->m_coords.j_low_bound() + range_t::jminus::value);
                // number of threads
            uint_t nx = (uint_t) (func_->m_coords.i_high_bound() + iplus - (func_->m_coords.i_low_bound() + iminus)+1);
            uint_t ny = (uint_t) (func_->m_coords.j_high_bound() + jplus - (func_->m_coords.j_low_bound() + jminus)+1);

//         uint_t nx_ = (uint_t) (func_->m_coords.i_high_bound() + std::abs(xrange_t::iplus) + std::abs(xrange_subdomain_t::iplus) - (func_->m_coords.i_low_bound() -std::abs(xrange_t::iminus) - std::abs(xrange_subdomain_t::iminus))/*+1*/);
//         uint_t ny_ = (uint_t) (func_->m_coords.j_high_bound() + std::abs(xrange_t::jplus) + std::abs(xrange_subdomain_t::jplus) - (func_->m_coords.j_low_bound() -std::abs(xrange_t::jminus) - std::abs(xrange_subdomain_t::jminus))/*+1*/);

            int ntx = 32, nty = 8, ntz = 1;
            dim3 threads(ntx, nty, ntz);

            int nbx = (nx + ntx - 1) / ntx;
            int nby = (ny + nty - 1) / nty;
            int nbz = 1;
            dim3 blocks(nbx, nby, nbz);

#ifndef NDEBUG
            printf("ntx = %d, nty = %d, ntz = %d\n",ntx, nty, ntz);
            printf("nbx = %d, nby = %d, nbz = %d\n",nbx, nby, nbz);
            printf("nx = %d, ny = %d, nz = 1\n",nx, ny);
#endif

            _impl_cuda::do_it_on_gpu<Arguments, EsfArguments, LocalDomain><<<blocks, threads>>>//<<<nbx*nby, ntx*nty>>>
                    (local_domain_gp,
                     coords_gp,
                     func_->m_coords.i_low_bound() + iminus,
                     func_->m_coords.j_low_bound() + jminus,
                     (nx),
                     (ny));
                cudaDeviceSynchronize();

            }
    };


} // namespace gridtools
