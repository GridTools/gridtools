#pragma once

namespace gridtools {
    /**
       This class implements the basic operation of iterating over a "column" in a interval.
    */
    // namespace _impl {
    //     struct inc {
    //         template <typename T>
    //         void operator()(T & x) const {
    //             ++x;
    //         }
    //     };
    // }

    namespace _impl {
        template <typename FunctorType,
                  typename IntervalMap,
                  typename LocalDomainType,
                  typename Coords>
        struct run_f_on_interval {
            Coords const &coords;
            LocalDomainType const &domain;

            GT_FUNCTION
            explicit run_f_on_interval(LocalDomainType & domain, Coords const& coords)
                : coords(coords)
                , domain(domain)
            {}

            template <typename Interval>
            GT_FUNCTION
            void operator()(Interval const&) const {
                typedef typename index_to_level<typename Interval::first>::type from;
                typedef typename index_to_level<typename Interval::second>::type to;
                if (boost::mpl::has_key<IntervalMap, Interval>::type::value) {
                    // printf("K Loop: %d ", coords.template value_at<from>());
                    // printf("-> %d\n", coords.template value_at<to>());

                    for (int k=coords.template value_at<from>(); k < coords.template value_at<to>(); ++k) {
                        typedef typename boost::mpl::at<IntervalMap, Interval>::type interval_type;
                        FunctorType::Do(domain, interval_type());
                        //boost::fusion::for_each(domain.local_iterators, _impl::inc());
                        domain.increment();
                    }
                }

            }
        };

    } // namespace _impl
} // namespace gridtools
