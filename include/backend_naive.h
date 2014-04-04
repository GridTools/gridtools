#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include "basic_token_execution.h"

namespace gridtools {
    namespace _impl_naive {

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

            template <typename Index>
            void operator()(Index const&) const {
                typedef typename boost::mpl::at<RangeSizes, Index>::type range_type;

                typedef typename boost::mpl::at<FunctorList, Index>::type functor_type;
                typedef typename boost::fusion::result_of::value_at<DomainList, Index>::type local_domain_type;

                typedef typename boost::mpl::at<FunctorsMap, Index>::type interval_map;

                local_domain_type& local_domain = boost::fusion::at<Index>(domain_list);

#ifndef NDEBUG
                std::cout << "Functor " << functor_type() << std::endl;
                std::cout << "I loop " << coords.i_low_bound() + range_type::iminus::value << " -> "
                          << coords.i_high_bound() + range_type::iplus::value << std::endl;
                std::cout << "J loop " << coords.j_low_bound() + range_type::jminus::value << " -> "
                          << coords.j_high_bound() + range_type::jplus::value << std::endl;
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
                std::cout << " ******************** " << first_hit() << std::endl;
                std::cout << " ******************** " << coords.template value_at<first_hit>() << std::endl;
#endif

                typedef typename local_domain_type::iterate_domain_type iterate_domain_type;

                for (int i = coords.i_low_bound() + range_type::iminus::value;
                     i < coords.i_high_bound() + range_type::iplus::value;
                     ++i)
                    for (int j = coords.j_low_bound() + range_type::jminus::value;
                         j < coords.j_high_bound() + range_type::jplus::value;
                         ++j) {
                        iterate_domain_type it_domain(local_domain, i,j, coords.template value_at<first_hit>()); 
                        //                        local_domain.move_to(i,j, coords.template value_at<first_hit>());
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

    struct backend_naive {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;

            gridtools::for_each<iter_range>(_impl_naive::run_functor
                                             <
                                             FunctorList,
                                             LoopIntervals,
                                             FunctorsMap,
                                             range_sizes,
                                             LocalDomainList,
                                             Coords
                                             >
                                             (local_domain_list,coords));
            std::cout << std::endl;
        }
    };

} // namespace gridtools
