#pragma once

namespace gridtools {
    /**
       This class implements the basic operation of iterating over a "column" in a interval.
    */
    namespace _impl {
        template <typename functor_type,
                  typename interval_map,
                  typename local_domain_type,
                  typename t_coords>
        struct run_f_on_interval {
            t_coords const &coords;
            local_domain_type const &domain;

            GT_FUNCTION
            explicit run_f_on_interval(local_domain_type & domain, t_coords const& coords)
                : coords(coords)
                , domain(domain)
            {}

            template <typename t_interval>
            GT_FUNCTION
            void operator()(t_interval const&) const {
                typedef typename index_to_level<typename t_interval::first>::type from;
                typedef typename index_to_level<typename t_interval::second>::type to;
                if (boost::mpl::has_key<interval_map, t_interval>::type::value) {
#ifndef __CUDA_ARCH__
#ifndef NDEBUG
                    std::cout << "K loop: " << coords.template value_at<from>() << " -> "
                              << coords.template value_at<to>() << std::endl;
#endif
#endif
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

    } // namespace _impl
} // namespace gridtools
