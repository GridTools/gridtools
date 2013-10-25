#pragma once

namespace _impl {
    template <typename t_interval, typename t_functor_list, typename t_functors_map, typename t_domain_list>
    struct run_functor
    {
        t_domain_list& domain_list;
        
        run_functor(t_domain_list& domain_list)
            : domain_list(domain_list)
        {}
        
        template<typename t_active_functor_index> 
        void operator()(t_active_functor_index) 
        { 
            // extract all functor information necessary
            typedef typename boost::mpl::at<t_functor_list, t_active_functor_index>::type Functor;
            typedef typename boost::mpl::at<typename boost::mpl::at<t_functors_map, t_active_functor_index>::type, t_interval>::type DoInterval;
//            typedef typename t_active_functor::first Functor;
//            typedef typename t_active_functor::second DoInterval;
//                        boost::mpl::pair<
//                            boost::mpl::at<t_functor_list, boost::mpl::_2>,
//                            boost::mpl::at<
//                                boost::mpl::at<t_functors_map, boost::mpl::_2>,
//                            t_interval
//                            >
//                        >
//                    >,

            Functor::Do(boost::fusion::at<t_active_functor_index>(domain_list), DoInterval()); 
            std::cout << "*" << Functor() << "* ";
        }
    };

    template <typename t_functor_list, 
              typename t_functors_map, 
              typename t_domain_list, 
              typename t_coords>
    struct show_info {
        t_coords const &coords;
        t_domain_list &domain_list;

        explicit show_info(t_domain_list & domain_list, t_coords const& coords)
            : coords(coords)
            , domain_list(domain_list)
        {}

        template <typename t_interval>
        void operator()(t_interval const&) const {
            typedef typename index_to_level<typename t_interval::first>::type from;
            typedef typename index_to_level<typename t_interval::second>::type to;
            std::cout << "{ (" << from() << " "
                      << to() << ") "
                      << "[" << coords.template value_at<from>() << ", "
                      << coords.template value_at<to>() << "] } ";
            
            typedef typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, boost::mpl::size<t_functor_list>::value>,
                boost::mpl::vector0<>,
                boost::mpl::if_<
                    boost::mpl::has_key<
                        boost::mpl::at<t_functors_map, boost::mpl::_2>,
                        t_interval
                    >,
                    boost::mpl::push_back<
                        boost::mpl::_1,
                        boost::mpl::_2>,
//                        boost::mpl::pair<
//                            boost::mpl::at<t_functor_list, boost::mpl::_2>,
//                            boost::mpl::at<
//                                boost::mpl::at<t_functors_map, boost::mpl::_2>,
//                            t_interval
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
                >(run_functor<t_interval, t_functor_list, t_functors_map, t_domain_list>(domain_list));
            }
        }
    };

}

struct backend0 {
    template <typename t_functor_list, // List of functors to execute (in order)
              typename range_sizes, // computed range sizes to know where to compute functot at<i>
              typename t_loop_intervals, // List of intervals on which functors are defined
              typename t_functors_map,  // Map between interval and actual arguments to pass to Do methods
              typename t_domain, // Domain class (not really useful maybe)
              typename t_coords, // Coordinate class with domain sizes and splitter coordinates
              typename t_local_domain_list> // List of local domain to be passed to functor at<i>
    static void run(t_domain const& domain, t_coords const& coords, t_local_domain_list &local_domain_list) {
        for (int i=coords.i_low_bound; i<coords.i_high_bound; ++i)
            for (int j=coords.j_low_bound; j<coords.j_high_bound; ++j) {
                std::cout << i << " " << j << " ";
                boost::mpl::for_each<t_loop_intervals>(_impl::show_info<t_functor_list,t_functors_map,t_local_domain_list,t_coords>(local_domain_list,coords));
                std::cout << std::endl;
            }
    }
};
