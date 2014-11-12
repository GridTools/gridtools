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
#include "../storage/cuda_storage.h"
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
        void do_it_on_gpu(typename Traits::local_domain_t * l_domain, typename Arguments::coords_t const* coords, uint_t starti, uint_t startj, uint_t nx, uint_t ny) {
            /* int i = blockIdx.x * blockDim.x + threadIdx.x; */
            /* int j = blockIdx.y * blockDim.y + threadIdx.y; */
	    uint_t z = coords->template value_at<typename Traits::first_hit_t>();

	    uint_t i = (blockIdx.x * blockDim.x + threadIdx.x)%ny;
	    uint_t j = (blockIdx.x * blockDim.x + threadIdx.x)/ny;

	    /* __shared__ float_type* m_data_pointer[Traits::local_domain_t::iterate_domain_t::N_DATA_POINTERS]; */

            if ((i < nx) && (j < ny)) {
	      typedef typename boost::mpl::front<typename Arguments::loop_intervals_t>::type interval;
	      typedef typename index_to_level<typename interval::first>::type from;
	      typedef typename index_to_level<typename interval::second>::type to;
	      typedef _impl::iteration_policy<from, to, Arguments::execution_type_t::type::iteration> iteration_policy;

	      typedef typename Traits::local_domain_t::iterate_domain_t iterate_domain_t;
	      typename Traits::iterate_domain_t it_domain(*l_domain, i+starti,j+startj/*, m_data_pointer*/,0,0);
	      //printf("setting the start to: %d \n",coords->template value_at< iteration_policy::from >() );	      //setting the initial k level (for backward/parallel iterations it is not 0)
	      if( !iteration_policy::value==enumtype::forward )
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


                /* struct extra_arguments{ */
                /*     typedef functor_type functor_t; */
                /*     typedef interval_map_type interval_map_t; */
                /*     typedef iterate_domain_t local_domain_t; */
                /*     typedef coords_type coords_t;}; */

#ifndef NDEBUG
                std::cout << "Functor " <<  functor_type() << "\n";
                std::cout << "I loop " << f->m_starti  + range_t::iminus::value << " -> "
                                    << f->m_starti + f->m_BI + range_t::iplus::value << "\n";
                std::cout << "J loop " << f->m_startj + range_t::jminus::value << " -> "
                                    << f->m_startj + f->m_BJ + range_t::jplus::value << "\n";
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


                local_domain.clone_to_gpu();
                f->m_coords.clone_to_gpu();

                local_domain_t *local_domain_gp = local_domain.gpu_object_ptr;

                coords_type const *coords_gp = f->m_coords.gpu_object_ptr;

                uint_t nx = f->m_coords.i_high_bound() + range_t::iplus::value - (f->m_coords.i_low_bound() + range_t::iminus::value);
                uint_t ny = f->m_coords.j_high_bound() + range_t::jplus::value - (f->m_coords.j_low_bound() + range_t::jminus::value);

                uint_t ntx = 8, nty = 32;//, ntz = 1;
                /* dim3 threads(ntx, nty, ntz); */

                ushort_t nbx = (nx + ntx - 1) / ntx;
                ushort_t nby = (ny + nty - 1) / nty;
                //ushort_t nbz = 1;
                /* dim3 blocks(nbx, nby, nbz); */

#ifndef NDEBUG
                printf("ntx = %d, nty = %d, ntz = %d\n",ntx, nty, ntz);
                printf("nbx = %d, nby = %d, nbz = %d\n",ntx, nty, ntz);
                printf("nx = %d, ny = %d, nz = 1\n",nx, ny);
#endif

		_impl_cuda::do_it_on_gpu<Arguments, Traits, extra_arguments<functor_type, interval_map_type, iterate_domain_t, coords_type> ><<<nbx*nby, ntx*nty>>>
                    (local_domain_gp,
                     coords_gp,
                     f->m_coords.i_low_bound() + range_t::iminus::value,
                     f->m_coords.j_low_bound() + range_t::jminus::value,
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
