#pragma once

#include <iostream>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/vector/vector0.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/fusion/include/at.hpp>

namespace gridtools {
    namespace _impl {
        template <typename Interval, typename FunctorList, typename FunctorsMap, typename DomainList>
        struct run_functor
        {
            DomainList& domain_list;
        
            run_functor(DomainList& domain_list)
                : domain_list(domain_list)
            {}
        
            template<typename ActiveFunctorIndex> 
            void operator()(ActiveFunctorIndex) 
            { 
                // extract all functor information necessary
                typedef typename boost::mpl::at<FunctorList, ActiveFunctorIndex>::type Functor;
                typedef typename boost::mpl::at<typename boost::mpl::at<FunctorsMap, ActiveFunctorIndex>::type, Interval>::type DoInterval;
                //            typedef typename ActiveFunctor::first Functor;
                //            typedef typename ActiveFunctor::second DoInterval;
                //                        boost::mpl::pair<
                //                            boost::mpl::at<FunctorList, boost::mpl::_2>,
                //                            boost::mpl::at<
                //                                boost::mpl::at<FunctorsMap, boost::mpl::_2>,
                //                            Interval
                //                            >
                //                        >
                //                    >,

                Functor::Do(boost::fusion::at<ActiveFunctorIndex>(domain_list), DoInterval()); 
                std::cout << "*" << Functor() << "* ";
            }
        };

        template <typename FunctorList, 
                  typename FunctorsMap, 
                  typename DomainList, 
                  typename Coords>
        struct show_info {
            Coords const &coords;
            DomainList &domain_list;

            explicit show_info(DomainList & domain_list, Coords const& coords)
                : coords(coords)
                , domain_list(domain_list)
            {}

            template <typename Interval>
            void operator()(Interval const&) const {
                typedef typename index_to_level<typename Interval::first>::type from;
                typedef typename index_to_level<typename Interval::second>::type to;
                std::cout << "{ (" << from() << " "
                          << to() << ") "
                          << "[" << coords.template value_at<from>() << ", "
                          << coords.template value_at<to>() << "] } ";
            
                typedef typename boost::mpl::fold<
                    boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::value>,
                    boost::mpl::vector0<>,
                    boost::mpl::if_<
                boost::mpl::has_key<
                boost::mpl::at<FunctorsMap, boost::mpl::_2>,
                    Interval
                    >,
                        boost::mpl::push_back<
                            boost::mpl::_1,
                            boost::mpl::_2>,
                        //                        boost::mpl::pair<
                        //                            boost::mpl::at<FunctorList, boost::mpl::_2>,
                        //                            boost::mpl::at<
                        //                                boost::mpl::at<FunctorsMap, boost::mpl::_2>,
                        //                            Interval
                        //                            >
                        //                        >
                        //                    >,
                        boost::mpl::_1
                        >
                >::type activefunctorsindexes;

        // for all active functors run their do methods
        for (int k=coords.template value_at<from>(); k < coords.template value_at<to>(); ++k) {
            std::cout << k << " ";
            boost::mpl::for_each<
            activefunctorsindexes
                >(run_functor<Interval, FunctorList, FunctorsMap, DomainList>(domain_list));
        }
    }
};

}

    struct backend0 {
        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be passed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {
            for (int i=coords.i_low_bound; i<coords.i_high_bound; ++i)
                for (int j=coords.j_low_bound; j<coords.j_high_bound; ++j) {
                    std::cout << i << " " << j << " ";
                    boost::mpl::for_each<LoopIntervals>(_impl::show_info<FunctorList,FunctorsMap,LocalDomainList,Coords>(local_domain_list,coords));
                    std::cout << std::endl;
                }
        }
    };
} // namespace gridtools
