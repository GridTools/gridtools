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
        template<>
	    struct _impl::cout<_impl::Cuda>
	    {
            const cout& operator  << (char* string) const
                {
                    printf(string);
                }
            const cout& operator << (int* number) const
                {
                    printf("%d", number);
                }
            const cout& operator << (float* number) const
                {
                    printf("%f", number);
                }

            template <typename T>
            const cout& operator << (T arg) const
                {
                    printf("You tried to print something");
                }

            //void endl() const {printf("%n");}
	    };

    }//namespace _impl

    namespace _impl_cuda {

        template <typename first_hit,
                  typename LoopIntervals,
                  typename functor_type,
                  typename interval_map,
                  typename LDomain,
                  typename Coords>
        __global__
        void do_it_on_gpu(LDomain * l_domain, Coords const* coords, int starti, int startj, int nx, int ny) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int z = coords->template value_at<first_hit>();

            if ((i < nx) && (j < ny)) {
                typedef typename LDomain::iterate_domain_type iterate_domain_type;
                iterate_domain_type it_domain(*l_domain, i+starti,j+startj, z);
                for_each<LoopIntervals>
                    (_impl::run_f_on_interval
                     <functor_type,
                     interval_map,
                     iterate_domain_type,
                     Coords>
                     (it_domain,*coords));
            }
        }

        /**
         * \brief this struct is the core of the ESF functor
         */
        template <typename FunctorList,
                  typename LoopIntervals,
                  typename FunctorsMap,
                  typename RangeSizes,
                  typename DomainList,
                  typename Coords>
	    struct run_functor_cuda : public _impl::run_functor <run_functor_cuda<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> >
        {

            typedef FunctorList functor_list_t;
            typedef LoopIntervals loop_intervals_t;
            typedef FunctorsMap functors_map_t;
            typedef RangeSizes range_sizes_t;
            typedef DomainList domain_list_t;
            typedef Coords coords_t;


            //\todo usful if we can use constexpr
            // static const _impl::BACKEND m_backend=_impl::Cuda;
            // static const _impl::BACKEND backend() {return m_backend;} //constexpr


// This struct maybe should't be inside another struct. I can put it elswhere and effectively use overloading in order to distinguish between the backends.
            template < typename Traits >
            struct execute_kernel_functor
            {
                typedef run_functor_cuda<FunctorList, LoopIntervals, FunctorsMap, RangeSizes, DomainList, Coords> backend_t;


                //template<_impl::STRATEGY s>
                static void wtf(){}

                template<_impl::STRATEGY s>
                static void execute_kernel( const typename Traits::local_domain_type& local_domain, const backend_t* f )
                // template < _impl::STRATEGY s, typename Traits >
                // void execute_kernel(  const typename Traits::local_domain_type& local_domain ) const
                    {
                        typedef typename Traits::range_type range_type;
                        typedef typename Traits::functor_type functor_type;
                        typedef typename Traits::local_domain_type  local_domain_type;
                        typedef typename Traits::interval_map interval_map;
                        typedef typename Traits::iterate_domain_type iterate_domain_type;

                        local_domain.clone_to_gpu();
                        f->coords.clone_to_gpu();

                        local_domain_type *local_domain_gp = local_domain.gpu_object_ptr;

                        Coords const *coords_gp = f->coords.gpu_object_ptr;

                        int nx = f->coords.i_high_bound() + range_type::iplus::value - (f->coords.i_low_bound() + range_type::iminus::value);
                        int ny = f->coords.j_high_bound() + range_type::jplus::value - (f->coords.j_low_bound() + range_type::jminus::value);

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
                        do_it_on_gpu<typename Traits::first_hit, LoopIntervals, functor_type, interval_map><<<blocks, threads>>>
                            (local_domain_gp,
                             coords_gp,
                             f->coords.i_low_bound() + range_type::iminus::value,
                             f->coords.j_low_bound() + range_type::jminus::value,
                             nx,
                             ny);
                        cudaDeviceSynchronize();

                    }

            };
            Coords const &coords;
            DomainList &domain_list;
        };

    }//namespace _impl_cuda


    namespace _impl
    {
//wasted code because of the lack of constexpr
        template <typename FunctorList,
                  typename LoopIntervals,
                  typename FunctorsMap,
                  typename RangeSizes,
                  typename DomainList,
                  typename Coords>
	    struct backend_type< _impl_cuda::run_functor_cuda<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> >
        {
            static const BACKEND m_backend=Cuda;
        };


/** traits struct defining the types chich are specific to the CUDA backend*/
        template<>
        struct backend_from_id<Cuda>
        {
            template <typename ValueType, typename Layout>
            struct storage_traits
            {
                typedef cuda_storage<ValueType, Layout> storage_type;
            };

            template <typename FunctorList,
                      typename LoopIntervals,
                      typename FunctorsMap,
                      typename RangeSizes,
                      typename DomainList,
                      typename Coords>
            struct execute_traits
            {
                typedef _impl_cuda::run_functor_cuda<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> run_functor;

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
