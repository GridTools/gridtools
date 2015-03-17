#pragma once

#include <stdio.h>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/value_at.hpp>

#include "execution_policy.h"
#include "heap_allocated_temps.h"
#include "../storage/hybrid_pointer.h"

#include "backend.h"
/**
 * @file
 * \brief implements the stencil operations for a GPU backend
 */

namespace gridtools {

/** Kernel function called from the GPU */
    namespace _impl_cuda {

        template <typename Arguments,
                  typename Traits,
                  typename ExtraArguments>
        __global__
        void do_it_on_gpu(typename Traits::local_domain_t const * __restrict__ l_domain, typename Arguments::coords_t const* coords, uint_t const starti, uint_t const startj, uint_t const nx, uint_t const ny) {
//             uint_t j = (blockIdx.x * blockDim.x + threadIdx.x)%ny;
//             uint_t i = (blockIdx.x * blockDim.x + threadIdx.x - j)/ny;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            uint_t z = coords->template value_at<typename Traits::first_hit_t>();

#ifndef NDEBUG
//             printf("index: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
//             printf("i and j : %d and %d\n", i, j);
#endif
            typedef typename Traits::local_domain_t::iterate_domain_t iterate_domain_t;

            __shared__
                void* data_pointer[iterate_domain_t::N_DATA_POINTERS];

            __shared__
                storage_cached<iterate_domain_t::N_STORAGES-1, typename Traits::local_domain_t::esf_args> strides;

            //Doing construction and assignment before the following 'if', so that we can
            //exploit parallel shared memory initialization
            typename Traits::iterate_domain_t it_domain(*l_domain);
            it_domain.template assign_storage_pointers<enumtype::Cuda>(data_pointer);

            it_domain.template assign_stride_pointers <enumtype::Cuda>(&strides);
            __syncthreads();

            if ((i < nx) && (j < ny)) {

#ifndef NDEBUG
//                printf("index_inside: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
//                printf("i  %d and j: %d inside\n", i, j);
#endif

                it_domain.template assign_ij<0>(i+starti,0);
                it_domain.template assign_ij<1>(j+startj,0);

                typedef typename boost::mpl::front<typename Arguments::loop_intervals_t>::type interval;
                typedef typename index_to_level<typename interval::first>::type from;
                typedef typename index_to_level<typename interval::second>::type to;
                typedef _impl::iteration_policy<from, to, Arguments::execution_type_t::type::iteration> iteration_policy;

                //printf("setting the start to: %d \n",coords->template value_at< iteration_policy::from >() );
                //setting the initial k level (for backward/parallel iterations it is not 0)
                if( !(iteration_policy::value==enumtype::forward) )
                    it_domain.set_k_start( coords->template value_at< iteration_policy::from >() );

                for_each<typename Arguments::loop_intervals_t>
                    (_impl::run_f_on_interval
                     <
                     typename Arguments::execution_type_t,
                     ExtraArguments
                     >
                     (it_domain,*coords));
            }

        }

        /**
         * \brief this struct is the core of the ESF functor
         */
        template < typename Arguments >
        struct run_functor_cuda : public _impl::run_functor < run_functor_cuda< Arguments > >
        {
            typedef _impl::run_functor < run_functor_cuda< Arguments > > super;
            explicit run_functor_cuda(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
                : super( domain_list, coords)
                {}

            explicit run_functor_cuda(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj)
                : super(domain_list, coords, i, j, bi, bj)
                {}

        };
    }//namespace _impl_cuda


//    namespace _impl
//    {
/** Partial specialization: naive implementation for the Cuda backend (2 policies specify strategy and backend)*/
    template < typename Arguments >
    struct execute_kernel_functor < _impl_cuda::run_functor_cuda<Arguments> >
    {
        typedef _impl_cuda::run_functor_cuda<Arguments> backend_t;

        template<
            typename FunctorType,
            typename IntervalMap,
            typename LocalDomainType,
            typename Coords>
        struct extra_arguments{
            typedef FunctorType functor_t;
            typedef IntervalMap interval_map_t;
            typedef LocalDomainType local_domain_t;
            typedef Coords coords_t;
        };

/**
   @brief core of the kernel execution
   \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
*/
        template < typename Traits >
        static void execute_kernel( typename Traits::local_domain_t& local_domain, const backend_t * f )
            {
                typedef typename Arguments::coords_t coords_type;
                // typedef typename Arguments::loop_intervals_t loop_intervals_t;
                typedef typename Traits::range_t range_t;
                typedef typename Traits::functor_t functor_type;
                typedef typename Traits::local_domain_t  local_domain_t;
                typedef typename Traits::interval_map_t interval_map_type;
                typedef typename Traits::iterate_domain_t iterate_domain_t;
                typedef typename Traits::first_hit_t first_hit_t;
                typedef typename Arguments::execution_type_t execution_type_t;

#ifdef CXX11_ENABLED
                typedef typename boost::mpl::eval_if_c<has_xrange<functor_type>::type::value, get_xrange< functor_type >, boost::mpl::identity<range<0,0,0> > >::type new_range_t;
                typedef typename sum_range<new_range_t, range_t>::type xrange_t;
                typedef typename functor_type::xrange_subdomain xrange_subdomain_t;
#else
                typedef typename sum_range<typename functor_type::xrange, range_t>::type xrange_t;
                typedef typename functor_type::xrange_subdomain xrange_subdomain_t;
#endif
                /* struct extra_arguments{ */
                /*     typedef functor_type functor_t; */
                /*     typedef interval_map_type interval_map_t; */
                /*     typedef iterate_domain_t local_domain_t; */
                /*     typedef coords_type coords_t;}; */


                local_domain.clone_to_gpu();
                f->m_coords.clone_to_gpu();

                local_domain_t *local_domain_gp = local_domain.gpu_object_ptr;

                coords_type const *coords_gp = f->m_coords.gpu_object_ptr;

                //generalize with flags?
                int boundary=f->m_coords.partitioner().boundary();
                int jminus=  xrange_subdomain_t::jminus::value + ((boundary)>7? xrange_t::jminus::value : 0) ;//j-low
                int iminus= xrange_subdomain_t::iminus::value + ((boundary%8)>3? xrange_t::iminus::value : 0) ;//i-low
                int jplus=  xrange_subdomain_t::jplus::value + ((boundary%4)>1? xrange_t::jplus::value : 0) ;//j-high
                int iplus= xrange_subdomain_t::iplus::value + ((boundary%2)>0? xrange_t::iplus::value : 0) ;//i-high

#ifndef NDEBUG
                int pid=0;
                MPI_Comm_rank(MPI_COMM_WORLD, &pid);

                std::cout<<"range< "<<xrange_subdomain_t::iminus::value<<","<<xrange_subdomain_t::iplus::value<<"..."<<std::endl;
                std::cout << "Boundary " <<  f->m_coords.partitioner().boundary() << "\n";
                std::cout << "Functor " <<  functor_type() << "\n";
                std::cout <<"[" << pid<< "] "<< "I loop " << f->m_starti<<"  + "<<iminus << " -> "
                          << f->m_starti<<" + "<<f->m_BI<<" + "<<iplus << "\n";
                std::cout <<"[" << pid<< "] "<< "J loop " << f->m_startj<<" + "<<jminus << " -> "
                          << f->m_startj<<" + "<<f->m_BJ<<" + "<<jplus << "\n";
                std::cout <<  " ******************** " << typename Traits::first_hit_t() << "\n";
                std::cout << " ******************** " << f->m_coords.template value_at<typename Traits::first_hit_t>() << "\n";

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

//         int nx = f->m_coords.i_high_bound() + range_t::iplus::value - (f->m_coords.i_low_bound() + range_t::iminus::value);
//         int ny = f->m_coords.j_high_bound() + range_t::jplus::value - (f->m_coords.j_low_bound() + range_t::jminus::value);
        // number of threads
        uint_t nx = (uint_t) (f->m_coords.i_high_bound() + iplus - (f->m_coords.i_low_bound() + iminus)+1);
        uint_t ny = (uint_t) (f->m_coords.j_high_bound() + jplus - (f->m_coords.j_low_bound() + jminus)+1);

//         uint_t nx_ = (uint_t) (f->m_coords.i_high_bound() + std::abs(xrange_t::iplus) + std::abs(xrange_subdomain_t::iplus) - (f->m_coords.i_low_bound() -std::abs(xrange_t::iminus) - std::abs(xrange_subdomain_t::iminus))/*+1*/);
//         uint_t ny_ = (uint_t) (f->m_coords.j_high_bound() + std::abs(xrange_t::jplus) + std::abs(xrange_subdomain_t::jplus) - (f->m_coords.j_low_bound() -std::abs(xrange_t::jminus) - std::abs(xrange_subdomain_t::jminus))/*+1*/);

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

                _impl_cuda::do_it_on_gpu<Arguments, Traits, extra_arguments<functor_type, interval_map_type, iterate_domain_t, coords_type> ><<<blocks, threads>>>//<<<nbx*nby, ntx*nty>>>
                    (local_domain_gp,
                     coords_gp,
                     f->m_coords.i_low_bound() + iminus,
                     f->m_coords.j_low_bound() + jminus,
                     (nx),
                     (ny));
                cudaDeviceSynchronize();

            }
    };

//    }//namespace _impl

/**@brief given the backend \ref gridtools::_impl_cuda::run_functor_cuda returns the backend ID gridtools::enumtype::Cuda
   wasted code because of the lack of constexpr*/
        template <typename Arguments>
        struct backend_type< _impl_cuda::run_functor_cuda<Arguments> >
        {
            static const enumtype::backend s_backend=enumtype::Cuda;
        };

} // namespace gridtools
