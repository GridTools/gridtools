#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/back.hpp>
#include "basic_token_execution.h"
#include "heap_allocated_temps.h"

namespace gridtools {
    namespace _impl_block {

        template <typename FunctorList,
                  typename LoopIntervals,
                  typename FunctorsMap,
                  typename RangeSizes,
                  typename DomainList,
                  typename Coords>
        struct run_functor_block {
            Coords const &coords;
            DomainList &domain_list;
            int starti, startj;
            const int BI, BJ;

            explicit run_functor_block(DomainList & domain_list, Coords const& coords,
                                  int starti, int startj,
                                  int BI, int BJ)
                : coords(coords)
                , domain_list(domain_list)
                , starti(starti)
                , startj(startj)
                , BI(BI)
                , BJ(BJ)
            {}

            template <typename Index>
            void operator()(Index const&) const {
                typedef typename boost::mpl::at<RangeSizes, Index>::type range_type;

                typedef typename boost::mpl::at<FunctorList, Index>::type functor_type;
                typedef typename boost::fusion::result_of::value_at<DomainList, Index>::type local_domain_type;

                typedef typename boost::mpl::at<FunctorsMap, Index>::type interval_map;

                local_domain_type& local_domain = boost::fusion::at<Index>(domain_list);

#ifndef NDEBUG
                std::cout << "Functor " << functor_type() << std::endl;
                std::cout << "I loop " << starti + range_type::iminus::value << " -> "
                          << starti + BI + range_type::iplus::value << std::endl;
                std::cout << "J loop " << startj + range_type::jminus::value << " -> "
                          << startj + BJ + range_type::jplus::value << std::endl;
#endif


                typedef typename index_to_level<typename boost::mpl::deref<typename boost::mpl::find_if<LoopIntervals, boost::mpl::has_key<interval_map, boost::mpl::_1> >::type>::type::first>::type first_hit;
#ifndef NDEBUG
                std::cout << " ******************** " << first_hit() << std::endl;
                std::cout << " ******************** " << coords.template value_at<first_hit>() << std::endl;
#endif

                typedef typename local_domain_type::iterate_domain_type iterate_domain_type;

                for (int i = starti + range_type::iminus::value;
                     i < starti + BI + range_type::iplus::value;
                     ++i)
                    for (int j = startj + range_type::jminus::value;
                         j < startj + BJ + range_type::jplus::value;
                         ++j) {
#ifndef NDEBUG
                        std::cout << "--------------------------------------"
                                  << i << ", "
                                  << " (" << startj << "," << j << ") " << j << ", "
                                  << coords.template value_at<first_hit>() << std::endl;
#endif
                        iterate_domain_type it_domain(local_domain, i,j, coords.template value_at<first_hit>());

                        gridtools::for_each<LoopIntervals>
                            (_impl::run_f_on_interval
                             <functor_type,
                             interval_map,
                             iterate_domain_type,
                             Coords>
                             (it_domain,coords)
                             );
                    }
            }

        };
    }


    struct backend_block: public heap_allocated_temps<backend_block> {
        static const int BI = 4;
        static const int BJ = 4;
        static const int BK = 0;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef storage<ValueType, Layout> type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary<storage<ValueType, Layout> > type;
        };

        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (maybe not really useful, local domain could be enoufg)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be passed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;

            typedef typename boost::mpl::at<range_sizes, typename boost::mpl::back<iter_range>::type >::type range_type;
            int n = coords.i_high_bound() + range_type::iplus::value - (coords.i_low_bound() + range_type::iminus::value);
            int m = coords.j_high_bound() + range_type::jplus::value - (coords.j_low_bound() + range_type::jminus::value);

            int NBI = n/BI;
            int NBJ = m/BJ;

#ifndef NDEBUG
            std::cout << "n = " << coords.i_high_bound() << " + "
                      << range_type::iplus::value << " - "
                      << "(" << coords.i_low_bound() << " + "
                      << range_type::iminus::value << ")"
                      << std::endl;
            std::cout << "m = " << coords.j_high_bound() << " + "
                      << range_type::jplus::value << " - "
                      << "(" << coords.j_low_bound() << " + "
                      << range_type::jminus::value << ")"
                      << std::endl;

            std::cout << "Iterations on i: " << n
                      << "\n"
                      << "Iteration on j: " << m
                      << std::endl;

            std::cout << "Number of blocks on i: " << NBI
                      << "\n"
                      << "Number of blocks on j: " << NBJ
                      << std::endl;
#endif

            for (int bi = 0; bi < NBI; ++bi) {
                for (int bj = 0; bj < NBJ; ++bj) {
                    int starti = bi*BI+coords.i_low_bound();
                    int startj = bj*BJ+coords.j_low_bound();
                    gridtools::for_each<iter_range>(_impl_block::run_functor_block
                                                     <
                                                     FunctorList,
                                                     LoopIntervals,
                                                     FunctorsMap,
                                                     range_sizes,
                                                     LocalDomainList,
                                                     Coords
                                                     >
                                                     (local_domain_list,coords,starti,startj, BI, BJ));
                }
            }

            for (int bj = 0; bj < NBJ; ++bj) {
                int starti = NBI*BI+coords.i_low_bound();
                int startj = bj*BJ+coords.j_low_bound();
                gridtools::for_each<iter_range>(_impl_block::run_functor_block
                                                <
                                                FunctorList,
                                                LoopIntervals,
                                                FunctorsMap,
                                                range_sizes,
                                                LocalDomainList,
                                                Coords
                                                >
                                                (local_domain_list,coords,starti,startj, n-NBI*BI, BJ));
            }

            for (int bi = 0; bi < NBI; ++bi) {
                int starti = bi*BI+coords.i_low_bound();
                int startj = NBJ*BJ+coords.j_low_bound();
                gridtools::for_each<iter_range>(_impl_block::run_functor_block
                                                <
                                                FunctorList,
                                                LoopIntervals,
                                                FunctorsMap,
                                                range_sizes,
                                                LocalDomainList,
                                                Coords
                                                >
                                                (local_domain_list,coords,starti,startj,BI, n-NBJ*BJ));
            }

            int starti = NBI*BI+coords.i_low_bound();
            int startj = NBJ*BJ+coords.j_low_bound();
            gridtools::for_each<iter_range>(_impl_block::run_functor_block
                                            <
                                            FunctorList,
                                            LoopIntervals,
                                            FunctorsMap,
                                            range_sizes,
                                            LocalDomainList,
                                            Coords
                                            >
                                            (local_domain_list,coords,starti,startj,n-NBI*BI,n-NBJ*BJ));

        }
    };
} // namespace gridtools
