#pragma once

#include <stdio.h>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/value_at.hpp>

#include "basic_token_execution.h"
#include "../storage/cuda_storage.h"
#include "heap_allocated_temps.h"

#include "backend.h"
/**
 * @file
 * \brief implements the stencil operations for a GPU backend
 */

namespace gridtools {

    namespace _impl{

/** Specialization of the cout function for the CUDA backend \TODO finish */
        template<>
	    struct cout<_impl::Cuda>
	    {
            const cout& operator  << (char* string) const
                {
                    printf(string);
                    return *this;
                }
            const cout& operator << (int* number) const
                {
                    printf("%d", number);
                    return *this;
                }
            const cout& operator << (float* number) const
                {
                    printf("%f", number);
                    return *this;
                }

            template <typename T>
            const cout& operator << (T arg) const
                {
                    printf("You tried to print something, and I don't know yet how to print it");
                    return *this;
                }
	    };

    }//namespace _impl

/** Kernel function called from the GPU */
    namespace _impl_cuda {

        template <typename FirstHit,
                  typename LoopIntervals,
                  typename FunctorType,
                  typename IntervalMap,
                  typename LDomain,
                  typename Coords>
        __global__
        void do_it_on_gpu(LDomain * l_domain, Coords const* coords, int starti, int startj, int nx, int ny) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int z = coords->template value_at<FirstHit>();

            if ((i < nx) && (j < ny)) {
                typedef typename LDomain::iterate_domain_t iterate_domain_t;
                iterate_domain_t it_domain(*l_domain, i+starti,j+startj, z);
                for_each<LoopIntervals>
                    (_impl::run_f_on_interval
                     <FunctorType,
                     IntervalMap,
                     iterate_domain_t,
                     Coords>
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

            explicit run_functor_cuda(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords, int i, int j, int bi, int bj)
                : super(domain_list, coords, i, j, bi, bj)
                {}

        };
    }//namespace _impl_cuda


    namespace _impl
    {
/** Partial specialization: naive implementation for the Cuda backend (2 policies specify strategy and backend)*/
    template < typename Arguments >
    struct execute_kernel_functor < _impl_cuda::run_functor_cuda<Arguments> >
    {
        typedef _impl_cuda::run_functor_cuda<Arguments> backend_t;

        template < typename Traits >
        static void execute_kernel( typename Traits::local_domain_t& local_domain, const backend_t * f )
            {
                typedef typename Arguments::coords_t coords_t;
                typedef typename Arguments::loop_intervals_t loop_intervals_t;
                typedef typename Traits::range_t range_t;
                typedef typename Traits::functor_t functor_t;
                typedef typename Traits::local_domain_t  local_domain_t;
                typedef typename Traits::interval_map_t interval_map_t;
                typedef typename Traits::iterate_domain_t iterate_domain_t;
                typedef typename Traits::first_hit_t first_hit_t;

                local_domain.clone_to_gpu();
                f->m_coords.clone_to_gpu();

                local_domain_t *local_domain_gp = local_domain.gpu_object_ptr;

                coords_t const *coords_gp = f->m_coords.gpu_object_ptr;

                int nx = f->m_coords.i_high_bound() + range_t::iplus::value - (f->m_coords.i_low_bound() + range_t::iminus::value);
                int ny = f->m_coords.j_high_bound() + range_t::jplus::value - (f->m_coords.j_low_bound() + range_t::jminus::value);

                int ntx = 8, nty = 32, ntz = 1;
                dim3 threads(ntx, nty, ntz);

                int nbx = (nx + ntx - 1) / ntx;
                int nby = (ny + nty - 1) / nty;
                int nbz = 1;
                dim3 blocks(nbx, nby, nbz);

#ifndef NDEBUG
                printf("ntx = %d, nty = %d, ntz = %d\n",ntx, nty, ntz);
                printf("nbx = %d, nby = %d, nbz = %d\n",ntx, nty, ntz);
                printf("nx = %d, ny = %d, nz = 1\n",nx, ny);
#endif
                _impl_cuda::do_it_on_gpu<first_hit_t, loop_intervals_t, functor_t, interval_map_t><<<blocks, threads>>>
                    (local_domain_gp,
                     coords_gp,
                     f->m_coords.i_low_bound() + range_t::iminus::value,
                     f->m_coords.j_low_bound() + range_t::jminus::value,
                     nx,
                     ny);
                cudaDeviceSynchronize();

            }
    };


///wasted code because of the lack of constexpr
        template <typename Arguments>
	    struct backend_type< _impl_cuda::run_functor_cuda<Arguments> >
        {
            static const BACKEND m_backend=Cuda;
        };


/** traits struct defining the types which are specific to the CUDA backend*/
        template<>
        struct backend_from_id< Cuda >
        {

            template <typename ValueType, typename Layout>
            struct storage_traits
            {
                typedef cuda_storage<ValueType, Layout> storage_t;
            };

            template <typename Arguments>
            struct execute_traits
            {
                typedef _impl_cuda::run_functor_cuda<Arguments> backend_t;
            };

            //function alias (pre C++11)
            template<
                typename Sequence
                , typename F
                >
            inline static void for_each(F f)
                {
                    boost::mpl::for_each<Sequence>(f);
                }

        };


    }//namespace _impl


} // namespace gridtools
