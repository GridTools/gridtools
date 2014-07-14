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

/**
 * @file
 * \brief implements the stencil operations for a GPU backend
 */

namespace gridtools {
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
        struct run_functor {
            Coords const &coords;
            DomainList &domain_list;

            explicit run_functor(DomainList & domain_list, Coords const& coords)
                : coords(coords)
                , domain_list(domain_list)
            {}

            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index const&) const {
                typedef typename boost::mpl::at<RangeSizes, Index>::type range_type;

                typedef typename boost::mpl::at<FunctorList, Index>::type functor_type;
                typedef typename boost::fusion::result_of::value_at<DomainList, Index>::type local_domain_type;

                typedef typename boost::mpl::at<FunctorsMap, Index>::type interval_map;

                local_domain_type &local_domain = boost::fusion::at<Index>(domain_list);

                local_domain.clone_to_gpu();
                coords.clone_to_gpu();

                local_domain_type *local_domain_gp = local_domain.gpu_object_ptr;

                Coords const *coords_gp = coords.gpu_object_ptr;

#ifndef NDEBUG
                local_domain.info();

                printf("I loop %d ", coords.i_low_bound() + range_type::iminus::value);
                printf("-> %d\n", coords.i_high_bound() + range_type::iplus::value);
                printf("J loop %d ", coords.j_low_bound() + range_type::jminus::value);
                printf("-> %d\n", coords.j_high_bound() + range_type::jplus::value);
#endif


                typedef typename index_to_level<
                    typename boost::mpl::deref<
                        typename boost::mpl::find_if<
                            LoopIntervals,
                            boost::mpl::has_key<interval_map, boost::mpl::_1>
                        >::type
                    >::type::first
                >::type first_hit;
#ifndef NDEBUG
                printf(" ********************\n", first_hit());
                printf(" ********************\n", coords.template value_at<first_hit>());
#endif

                int nx = coords.i_high_bound() + range_type::iplus::value - (coords.i_low_bound() + range_type::iminus::value);
                int ny = coords.j_high_bound() + range_type::jplus::value - (coords.j_low_bound() + range_type::jminus::value);

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

                do_it_on_gpu<first_hit, LoopIntervals, functor_type, interval_map><<<blocks, threads>>>
                    (local_domain_gp, 
                     coords_gp, 
                     coords.i_low_bound() + range_type::iminus::value, 
                     coords.j_low_bound() + range_type::jminus::value, 
                     nx, 
                     ny);
                cudaDeviceSynchronize();

            //     for (int i = coords.i_low_bound() + range_type::iminus::value;
            //          i < coords.i_high_bound() + range_type::iplus::value;
            //          ++i)
            //         for (int j = coords.j_low_bound() + range_type::jminus::value;
            //              j < coords.j_high_bound() + range_type::jplus::value;
            //              ++j) {
            //             local_domain.move_to(i,j, coords.template value_at<first_hit>());
            //             for_each<LoopIntervals>(_impl::run_f_on_interval<functor_type, interval_map,local_domain_type,Coords>(local_domain,coords));
            //         }
            }

        };
    }

    struct backend_cuda: public heap_allocated_temps<backend_cuda> {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef cuda_storage<ValueType, Layout> type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary<cuda_storage<ValueType, Layout> > type;
        };

        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * \tparam FunctorList  List of functors to execute (in order)
         * \tparam range_sizes computed range sizes to know where to compute functot at<i>
         * \tparam LoopIntervals List of intervals on which functors are defined
         * \tparam FunctorsMap Map between interval and actual arguments to pass to Do methods
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Coords Coordinate class with domain sizes and splitter coordinates
         * \tparam LocalDomainList List of local domain to be pbassed to functor at<i>
         */
        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;

            boost::mpl::for_each<iter_range>(_impl_cuda::run_functor
                                             <
                                             FunctorList,
                                             LoopIntervals,
                                             FunctorsMap,
                                             range_sizes,
                                             LocalDomainList,
                                             Coords
                                             >
                                             (local_domain_list,coords));
        }
    };

} // namespace gridtools
