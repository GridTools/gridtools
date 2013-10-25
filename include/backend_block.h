#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/back.hpp>

namespace _impl {

    template <typename functor_type,
              typename interval_map,
              typename local_domain_type,
              typename t_coords>
    struct run_f_on_interval {
        t_coords const &coords;
        local_domain_type &domain;

        explicit run_f_on_interval(local_domain_type & domain, t_coords const& coords)
            : coords(coords)
            , domain(domain)
        {}

        template <typename t_interval>
        void operator()(t_interval const&) const {
            typedef typename index_to_level<typename t_interval::first>::type from;
            typedef typename index_to_level<typename t_interval::second>::type to;
            if (boost::mpl::has_key<interval_map, t_interval>::type::value) {
                std::cout << "K loop: " << coords.template value_at<from>() << " -> "
                          << coords.template value_at<to>() << std::endl; 
                for (int k=coords.template value_at<from>(); k < coords.template value_at<to>(); ++k) {
                    //std::cout << k << " yessssssss ";
                    //                    int a = typename boost::mpl::at<interval_map, t_interval>::type();
                    typedef typename boost::mpl::at<interval_map, t_interval>::type interval_type;
                    functor_type::Do(domain, interval_type());
                    domain.increment();
                }
            }

        }
    };

    template <typename t_functor_list, 
              typename t_loop_intervals, 
              typename t_functors_map, 
              typename t_range_sizes, 
              typename t_domain_list, 
              typename t_coords, int BI, int BJ>
    struct run_functor {
        t_coords const &coords;
        t_domain_list &domain_list;
        int starti, startj;

        explicit run_functor(t_domain_list & domain_list, t_coords const& coords, int starti, int startj)
            : coords(coords)
            , domain_list(domain_list)
            , starti(starti)
            , startj(startj)
        {}

        template <typename t_index>
        void operator()(t_index const&) const {
            typedef typename boost::mpl::at<t_range_sizes, t_index>::type range_type;

            typedef typename boost::mpl::at<t_functor_list, t_index>::type functor_type;
            typedef typename boost::fusion::result_of::value_at<t_domain_list, t_index>::type local_domain_type;

            typedef typename boost::mpl::at<t_functors_map, t_index>::type interval_map;

            local_domain_type local_domain = boost::fusion::at<t_index>(domain_list);

            std::cout << "Functor " << functor_type() << std::endl;
            std::cout << "I loop " << starti + range_type::iminus::value << " -> "
                      << starti + BI + range_type::iplus::value << std::endl;
            std::cout << "J loop " << startj + range_type::jminus::value << " -> "
                      << startj + BJ + range_type::jplus::value << std::endl;


            typedef typename index_to_level<typename boost::mpl::deref<typename boost::mpl::find_if<t_loop_intervals, boost::mpl::has_key<interval_map, boost::mpl::_1> >::type>::type::first>::type first_hit;
            std::cout << " ******************** " << first_hit() << std::endl;
            std::cout << " ******************** " << coords.template value_at<first_hit>() << std::endl;

            for (int i = starti + range_type::iminus::value;
                 i < starti + BI + range_type::iplus::value;
                 ++i)
                for (int j = startj + range_type::jminus::value;
                     j < startj + BJ + range_type::jplus::value;
                     ++j) {
                    std::cout << "--------------------------------------" 
                              << i << ", " 
                              << " (" << startj << "," << j << ") " << j << ", " 
                              << coords.template value_at<first_hit>() << std::endl;                    
                    local_domain.move_to(i,j, coords.template value_at<first_hit>());
                    boost::mpl::for_each<t_loop_intervals>(run_f_on_interval<functor_type, interval_map,local_domain_type,t_coords>(local_domain,coords));
                }
        }

    };
}

struct backend_block {
    template <typename t_functor_list, // List of functors to execute (in order)
              typename range_sizes, // computed range sizes to know where to compute functot at<i>
              typename t_loop_intervals, // List of intervals on which functors are defined
              typename t_functors_map,  // Map between interval and actual arguments to pass to Do methods
              typename t_domain, // Domain class (maybe not really useful, local domain could be enoufg)
              typename t_coords, // Coordinate class with domain sizes and splitter coordinates
              typename t_local_domain_list> // List of local domain to be passed to functor at<i>
    static void run(t_domain const& domain, t_coords const& coords, t_local_domain_list &local_domain_list) {

        typedef typename boost::mpl::range_c<int, 0, boost::mpl::size<t_functor_list>::type::value> iter_range;

        static const int BI = 2;
        static const int BJ = 2;

        typedef typename boost::mpl::at<range_sizes, typename boost::mpl::back<iter_range>::type >::type range_type;
        int n = coords.i_high_bound + range_type::iplus::value - (coords.i_low_bound + range_type::iminus::value);
        int m = coords.j_high_bound + range_type::jplus::value - (coords.j_low_bound + range_type::jminus::value);

        int NBI = n/BI;
        int NBJ = m/BJ;
        std::cout << "n = " << coords.i_high_bound << " + " 
                  << range_type::iplus::value << " - " 
                  << "(" << coords.i_low_bound << " + "
                  << range_type::iminus::value << ")"
                  << std::endl;
        std::cout << "m = " << coords.j_high_bound << " + " 
                  << range_type::jplus::value << " - " 
                  << "(" << coords.j_low_bound << " + "
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

        for (int bi = 0; bi < NBI; ++bi) {
            int starti = bi*BI+coords.i_low_bound;
            for (int bj = 0; bj < NBJ; ++bj) {
                int startj = bj*BJ+coords.j_low_bound;
                boost::mpl::for_each<iter_range>(_impl::run_functor
                                                 <
                                                 t_functor_list,
                                                 t_loop_intervals,
                                                 t_functors_map,
                                                 range_sizes,
                                                 t_local_domain_list,
                                                 t_coords, BI, BJ
                                                 >
                                                 (local_domain_list,coords,starti,startj));
            }
        }
        std::cout << std::endl;
    }
};
