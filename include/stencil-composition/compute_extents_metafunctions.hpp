#pragma once
#include <boost/mpl/fold.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>

#include "./esf_metafunctions.hpp"
#include "./wrap_type.hpp"
#include "./mss.hpp"
#include "./amss_descriptor.hpp"
#include "./mss_metafunctions.hpp"
#include "./reductions/reduction_descriptor.hpp"
#include "./linearize_mss_functions.hpp"

namespace gridtools {

    template < typename PlaceholdersVector, typename InitExtent = extent<> >
    struct init_map_of_extents {
        typedef typename boost::mpl::fold< PlaceholdersVector,
            boost::mpl::map0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::pair< boost::mpl::_2, InitExtent > > >::type type;
    };

    template < typename PlaceholdersMap >
    struct compute_extents_of {

        template < typename MssDescriptor >
        struct for_mss {
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< MssDescriptor >::value or MssDescriptor::is_reduction_t::value),
                "Internal Error: invalid type");
            template < typename CurrentRange >
            struct work_on {
                template < typename PlcRangePair, typename CurrentMap >
                struct with {
                    typedef typename sum_extent< CurrentRange, typename PlcRangePair::second >::type candidate_extent;
                    typedef typename enclosing_extent< candidate_extent,
                        typename boost::mpl::at< CurrentMap, typename PlcRangePair::first >::type >::type extent;
                    typedef typename boost::mpl::erase_key< CurrentMap, typename PlcRangePair::first >::type map_erased;
                    typedef typename boost::mpl::insert< map_erased,
                        boost::mpl::pair< typename PlcRangePair::first, extent > >::type type; // new map
                };
            };

            template < typename Output, typename Inputs, typename CurrentMap >
            struct for_each_output {
                typedef typename boost::mpl::at< CurrentMap, typename Output::first >::type current_extent;

                typedef typename boost::mpl::fold< Inputs,
                    CurrentMap,
                    typename work_on< current_extent >::template with< boost::mpl::_2, boost::mpl::_1 > >::type
                    type; // the new map
            };

            template < typename ESFs, typename CurrentMap, int Elements >
            struct populate_map {
                typedef typename boost::mpl::at_c< ESFs, 0 >::type current_ESF;
                typedef typename boost::mpl::pop_front< ESFs >::type rest_of_ESFs;

                typedef typename esf_get_w_per_functor< current_ESF, boost::true_type >::type outputs;
                GRIDTOOLS_STATIC_ASSERT((check_all_extents_are< outputs, extent<> >::type::value),
                    "Extents of the outputs of ESFs are not all empty. All outputs must have empty extents");

                typedef typename esf_get_r_per_functor< current_ESF, boost::true_type >::type inputs;

                typedef typename boost::mpl::fold< outputs,
                    CurrentMap,
                    for_each_output< boost::mpl::_2, inputs, boost::mpl::_1 > >::type new_map;

                typedef
                    typename populate_map< rest_of_ESFs, new_map, boost::mpl::size< rest_of_ESFs >::type::value >::type
                        type;
            };

            template < typename ESFs, typename CurrentMap >
            struct populate_map< ESFs, CurrentMap, 0 > {
                typedef CurrentMap type;
            };

            typedef typename boost::mpl::reverse< typename unwrap_independent<
                typename mss_descriptor_esf_sequence< MssDescriptor >::type >::type >::type ESFs;

            typedef typename populate_map< ESFs, PlaceholdersMap, boost::mpl::size< ESFs >::type::value >::type type;
        }; // struct for_mss
    };     // struct compute_extents_of
}
