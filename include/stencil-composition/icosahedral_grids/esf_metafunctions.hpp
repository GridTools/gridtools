#pragma once
#include <boost/mpl/equal.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/set/set0.hpp>
#include "../../common/generic_metafunctions/is_there_in_sequence_if.hpp"

namespace gridtools {
    namespace icgrid {
        template < typename EsfSequence >
        struct extract_esf_location_type {
            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Error, wrong esf types");
            typedef typename boost::mpl::fold< EsfSequence,
                boost::mpl::set0<>,
                boost::mpl::insert< boost::mpl::_1, esf_get_location_type< boost::mpl::_2 > > >::type
                location_type_set_t;
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< location_type_set_t >::value == 1),
                "Error: all ESFs should have the same location type");

            typedef typename boost::mpl::front< location_type_set_t >::type type;
        };

        /**
         * metafunction that returns true if: "at least one esf of the sequence has a color that matches
         * the Color parameter or the color of the esf is specified as nocolor
         * (meaning that any color should be matched)
         * @tparam EsfSequence sequence of esfs
         * @tparam Color color to be matched by the ESFs
         */
        template<typename EsfSequence, typename Color>
        struct esf_sequence_contains_color
        {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), "Error");
            GRIDTOOLS_STATIC_ASSERT((is_color_type<Color>::value), "Error");

            template<typename Esf>
            struct esf_has_color_{
                GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Error");
                typedef static_bool<
                    (boost::is_same< typename Esf::color_t::color_t, typename Color::color_t >::value ||
                boost::is_same<typename Esf::color_t, nocolor>::value) > type;
            };

            typedef typename is_there_in_sequence_if<EsfSequence, esf_has_color_<boost::mpl::_> >::type type;
        };

    }

    template < typename Esf1, typename Esf2 >
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT(
            (is_esf_descriptor< Esf1 >::value && is_esf_descriptor< Esf2 >::value), "Error: Internal Error");
        typedef static_bool< boost::is_same< typename Esf1::esf_function, typename Esf2::esf_function >::value &&
                             boost::mpl::equal< typename Esf1::args_t, typename Esf2::args_t >::value &&
                             boost::is_same< typename Esf1::location_type, typename Esf2::location_type >::value &&
                             boost::is_same< typename Esf1::grid_t, typename Esf2::grid_t >::value > type;
    };

    struct extract_esf_functor {
        template < typename Esf >
        struct apply {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Error");

            typedef typename Esf::template esf_function<0> type;
        };
    };

    template<typename Esf>
    struct esf_arg_list
    {
        template<typename Set, typename Item>
        struct insert_arglist
        {
            typedef typename boost::mpl::insert<Set, typename Esf::template esf_function<Item::value>::arg_list >::type type;
        };

        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Error");

        typedef typename boost::mpl::fold<
            boost::mpl::range_c<uint_t, 0, Esf::location_type::n_colors::value>,
            boost::mpl::set0<>,
            insert_arglist<boost::mpl::_1, boost::mpl::_2>
        >::type checkset;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<checkset>::value==1), "Multiple Color specializations of the same ESF must contain the same arg list");

        typedef typename Esf::template esf_function<0>::arg_list type;
    };

} // namespace gridtools
