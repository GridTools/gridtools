#include <boost/mpl/and.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/contains.hpp>

namespace gridtools {

    namespace _impl {
        template <typename Aggregator>
        struct investigate_esf {
            template <typename Agg>
            struct investigate_placeholder {
                template <typename CR, typename Plc>
                struct apply {
                    using type = typename boost::mpl::and_<CR, typename boost::mpl::contains<typename Agg::placeholders, Plc>::type>::type;
                };
            };

            template <typename CurrentResult, typename ESF>
            struct apply {
                using type = typename boost::mpl::fold<
                    typename ESF::args_t,
                    CurrentResult,
                    typename investigate_placeholder<Aggregator>::template apply<boost::mpl::_1, boost::mpl::_2>
                    >::type;
            };
        };

        template <typename Aggregator, typename... RestOfMss>
        struct unwrap_esf_sequence;

        template <typename CurrentResult, typename Aggregator, typename FirstMss, typename... RestOfMss>
        struct unwrap_esf_sequence<CurrentResult, Aggregator, FirstMss, RestOfMss...> {
            using esfs = typename FirstMss::esf_sequence_t;
            using type = typename boost::mpl::fold<
                esfs,
                CurrentResult,
                typename investigate_esf<Aggregator>::template apply<boost::mpl::_1, boost::mpl::_2>
                >::type;
        };

        // SHORT CIRCUITING THE AND
        template <typename Aggregator, typename... RestOfMss>
        struct unwrap_esf_sequence<boost::mpl::false_, Aggregator, RestOfMss...> {
            using type = boost::mpl::false_;
        };

        // Recursion base
        template <typename Aggregator>
        struct unwrap_esf_sequence<Aggregator> {
            using type = boost::mpl::true_;
        };

        /**
           This metafuction is for debugging purpose. It checks that
           all the pplaceholders used in the making of a computation
           are also listed in the aggregator.
        */
        template <typename Aggregator, typename... Mss>
        struct all_args_in_aggregator {
            using type = typename unwrap_esf_sequence<boost::mpl::true_, Aggregator, Mss...>::type;
        }; // struct all_args_in_domain
    } // namespace _impl
} // namespace gridtools
