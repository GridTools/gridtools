#pragma once
#include <boost/mpl/fold.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/range_c.hpp>

#include "stencil-composition/wrap_type.hpp"
#include "../mss.hpp"
#include "../amss_descriptor.hpp"
#include "../mss_metafunctions.hpp"

namespace gridtools {

    namespace strgrid {

        template < typename FunctorDesc >
        struct extract_extents {
            typedef typename FunctorDesc::esf_function Functor;

            /**@brief here the extents for the functors are calculated: the resulting type will be the extent (i,j)
             * which is enclosing all the extents of the field used by the specific functor*/
            template < typename RangeState, typename ArgumentIndex >
            struct update_extent {
                typedef typename boost::mpl::at< typename Functor::arg_list, ArgumentIndex >::type argument_type;
                typedef typename enclosing_extent< RangeState, typename argument_type::extent_t >::type type;
            };

            /**@brief here the extents for the functors are calculated: iterates over the fields and calls the
             * metafunction above*/
            typedef typename boost::mpl::fold<
                boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename Functor::arg_list >::type::value >,
                extent< 0, 0, 0, 0, 0, 0 >,
                update_extent< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        template < typename NotIndependentElem >
        struct from_independents {
            typedef boost::false_type type;
        };

        /**@brief specialization for "independent" elementary stencil functions: given the list of  functors inside an
         * elementary stencil function (esf) returns a vector of enclosing extents, one per functor*/
        template < typename T >
        struct from_independents< independent_esf< T > > {
            typedef typename boost::mpl::fold< typename independent_esf< T >::esf_list,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1, extract_extents< boost::mpl::_2 > > >::type raw_type;

            typedef _impl::wrap_type< raw_type > type;
        };

        template < typename T >
        struct extract_extents< independent_esf< T > > {
            typedef boost::false_type type;
        };

        /** @brief metafunction returning, given the elementary stencil function "Elem", either the vector of enclosing
         * extents (in case of "independent" esf), or the single extent enclosing all the extents. */
        template < typename State, typename Elem >
        struct traverse_extents {
            typedef typename boost::mpl::push_back< State,
                typename boost::mpl::if_< is_independent< Elem >,
                                                        typename from_independents< Elem >::type,
                                                        typename extract_extents< Elem >::type >::type >::type type;
        };

        /**@brief prefix sum, scan operation, takes into account the extent needed by the current stage plus the extent
         * needed by the next stage.*/
        template < typename ListOfRanges >
        struct prefix_on_extents {

            template < typename List, typename Range /*, typename NextRange*/ >
            struct state {
                typedef List list;
                typedef Range extent;
                // typedef NextRange next_extent;
            };

            template < typename PreviousState, typename CurrentElement >
            struct update_state {
                typedef typename sum_extent< typename PreviousState::extent, CurrentElement >::type new_extent;
                typedef typename boost::mpl::push_front< typename PreviousState::list,
                    typename PreviousState::extent >::type new_list;
                typedef state< new_list, new_extent > type;
            };

            template < typename PreviousState, typename IndVector >
            struct update_state< PreviousState, _impl::wrap_type< IndVector > > {
                typedef typename boost::mpl::fold<
                    IndVector,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back< boost::mpl::_1,
                        /*sum_extent<*/ typename PreviousState::extent /*, boost::mpl::_2>*/ > >::type raw_extents;

                typedef typename boost::mpl::fold< IndVector,
                    extent< 0, 0, 0, 0, 0, 0 >,
                    enclosing_extent< boost::mpl::_1, sum_extent< typename PreviousState::extent, boost::mpl::_2 > > >::
                    type final_extent;

                typedef typename boost::mpl::push_front< typename PreviousState::list,
                    _impl::wrap_type< raw_extents > >::type new_list;

                typedef state< new_list, final_extent > type;
            };

            typedef typename boost::mpl::reverse_fold< ListOfRanges,
                state< boost::mpl::vector0<>, extent< 0, 0, 0, 0, 0, 0 > >,
                update_state< boost::mpl::_1, boost::mpl::_2 > >::type final_state;

            typedef typename final_state::list type;
        };

        template < typename State, typename SubArray >
        struct keep_scanning {
            typedef typename boost::mpl::fold< typename SubArray::type,
                State,
                boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        template < typename Array >
        struct linearize_extent_sizes {
            typedef
                typename boost::mpl::fold< Array,
                    boost::mpl::vector0<>,
                    boost::mpl::if_< _impl::is_wrap_type< boost::mpl::_2 >,
                                               keep_scanning< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > > >::type type;
        };

        template < typename MssDescriptor >
        struct mss_compute_extent_sizes {
            GRIDTOOLS_STATIC_ASSERT((is_computation_token< MssDescriptor >::value), "Internal Error: invalid type");

            /**
             * \brief Here the extents are calculated recursively, in order for each functor's domain to embed all the
             * domains of the functors he depends on.
             */
            typedef typename boost::mpl::fold< typename mss_descriptor_esf_sequence< MssDescriptor >::type,
                boost::mpl::vector0<>,
                traverse_extents< boost::mpl::_1, boost::mpl::_2 > >::type extents_list;

            /*
             *  Compute prefix sum to compute bounding boxes for calling a given functor
             */
            typedef typename prefix_on_extents< extents_list >::type structured_extent_sizes;

            /**
             * linearize the data flow graph
             *
             */
            typedef typename linearize_extent_sizes< structured_extent_sizes >::type type;

            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type >::value ==
                    boost::mpl::size< type >::value),
                "Internal Error: wrong size");
        };

        template < typename Mss1, typename Mss2, typename Cond >
        struct mss_compute_extent_sizes< condition< Mss1, Mss2, Cond > > {
            typedef condition< typename mss_compute_extent_sizes< Mss1 >::type,
                typename mss_compute_extent_sizes< Mss1 >::type,
                Cond > type;
        };

        template < typename Placeholders >
        struct compute_extents_of {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Placeholders, is_arg >::value), "wrong type");
            template < typename MssDescriptor >
            struct for_mss {
                GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< MssDescriptor >::value), "Internal Error: invalid type");

                template < typename PLH >
                struct map_of_empty_extents {
                    typedef typename boost::mpl::fold<
                        PLH,
                        boost::mpl::map0<>,
                        boost::mpl::insert< boost::mpl::_1,
                            boost::mpl::pair< boost::mpl::_2, extent< 0, 0, 0, 0, 0, 0 > > > >::type type;
                };

                template < typename CurrentRange >
                struct work_on {
                    template < typename PlcRangePair, typename CurrentMap >
                    struct with {
                        typedef
                            typename sum_extent< CurrentRange, typename PlcRangePair::second >::type candidate_extent;
                        typedef typename enclosing_extent< candidate_extent,
                            typename boost::mpl::at< CurrentMap, typename PlcRangePair::first >::type >::type extent;
                        typedef
                            typename boost::mpl::erase_key< CurrentMap, typename PlcRangePair::first >::type map_erased;
                        typedef typename boost::mpl::insert< map_erased,
                            boost::mpl::pair< typename PlcRangePair::first, extent > >::type type; // new map
                    };
                };

                template < typename ESFs, typename CurrentMap, int Elements >
                struct populate_map {
                    typedef typename boost::mpl::at_c< ESFs, 0 >::type current_ESF;
                    typedef typename boost::mpl::pop_front< ESFs >::type rest_of_ESFs;

                    typedef typename esf_get_the_only_w_per_functor< current_ESF, boost::false_type >::type output;
                    // ^^^^ they (must) have the same extent<0,0,0,0,0,0> [so not need for true predicate]
                    // now assuming there is only one

                    typedef typename esf_get_r_per_functor< current_ESF, boost::true_type >::type inputs;

                    typedef typename boost::mpl::at< CurrentMap, output >::type current_extent;

                    typedef typename boost::mpl::fold< inputs,
                        CurrentMap,
                        typename work_on< current_extent >::template with< boost::mpl::_2, boost::mpl::_1 > >::type
                        new_map;

                    typedef typename populate_map< rest_of_ESFs,
                        new_map,
                        boost::mpl::size< rest_of_ESFs >::type::value >::type type;
                };

                template < typename ESFs, typename CurrentMap >
                struct populate_map< ESFs, CurrentMap, 0 > {
                    typedef CurrentMap type;
                };

                typedef
                    typename boost::mpl::reverse< typename mss_descriptor_esf_sequence< MssDescriptor >::type >::type
                        ESFs;

                typedef typename populate_map< ESFs,
                    typename map_of_empty_extents< Placeholders >::type,
                    boost::mpl::size< ESFs >::type::value >::type type;
            };
        };
    }
}
