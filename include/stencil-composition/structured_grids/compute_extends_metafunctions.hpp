#pragma once
#include "stencil-composition/wrap_type.hpp"

namespace gridtools {

    namespace strgrid {

        template <typename FunctorDesc>
        struct extract_extends {
            typedef typename FunctorDesc::esf_function Functor;

            /**@brief here the extends for the functors are calculated: the resulting type will be the extend (i,j) which is enclosing all the extends of the field used by the specific functor*/
            template <typename RangeState, typename ArgumentIndex>
            struct update_extend {
                typedef typename boost::mpl::at<typename Functor::arg_list, ArgumentIndex>::type argument_type;
                typedef typename enclosing_extend<RangeState, typename argument_type::extend_type>::type type;
            };

            /**@brief here the extends for the functors are calculated: iterates over the fields and calls the metafunction above*/
            typedef typename boost::mpl::fold<
                boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename Functor::arg_list>::type::value >,
                extend<0,0,0,0,0,0>,
                update_extend<boost::mpl::_1, boost::mpl::_2>
                >::type type;
        };

        template <typename NotIndependentElem>
        struct from_independents {
            typedef boost::false_type type;
        };

        /**@brief specialization for "independent" elementary stencil functions: given the list of  functors inside an elementary stencil function (esf) returns a vector of enclosing extends, one per functor*/
        template <typename T>
        struct from_independents<independent_esf<T> > {
            typedef typename boost::mpl::fold<
                typename independent_esf<T>::esf_list,
                boost::mpl::vector0<>,
                boost::mpl::push_back<boost::mpl::_1, extract_extends<boost::mpl::_2> >
            >::type raw_type;

            typedef _impl::wrap_type<raw_type> type;
        };

        template <typename T>
        struct extract_extends<independent_esf<T> >
        {
            typedef boost::false_type type;
        };


        /** @brief metafunction returning, given the elementary stencil function "Elem", either the vector of enclosing extends (in case of "independent" esf), or the single extend enclosing all the extends. */
        template <typename State, typename Elem>
        struct traverse_extends {
            typedef typename boost::mpl::push_back<
                State,
                typename boost::mpl::if_<
                    is_independent<Elem>,
                    typename from_independents<Elem>::type,
                    typename extract_extends<Elem>::type
                >::type
            >::type type;
        };


        /**@brief prefix sum, scan operation, takes into account the extend needed by the current stage plus the extend needed by the next stage.*/
        template <typename ListOfRanges>
        struct prefix_on_extends {

            template <typename List, typename Range/*, typename NextRange*/>
            struct state {
                typedef List list;
                typedef Range extend;
                // typedef NextRange next_extend;
            };

            template <typename PreviousState, typename CurrentElement>
            struct update_state {
                typedef typename sum_extend<typename PreviousState::extend,
                                               CurrentElement>::type new_extend;
                typedef typename boost::mpl::push_front<typename PreviousState::list, typename PreviousState::extend>::type new_list;
                typedef state<new_list, new_extend> type;
            };

            template <typename PreviousState, typename IndVector>
            struct update_state<PreviousState, _impl::wrap_type<IndVector> >
            {
                typedef typename boost::mpl::fold<
                    IndVector,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back<boost::mpl::_1, /*sum_extend<*/typename PreviousState::extend/*, boost::mpl::_2>*/ >
                >::type raw_extends;

                typedef typename boost::mpl::fold<
                    IndVector,
                    extend<0,0,0,0,0,0>,
                    enclosing_extend<boost::mpl::_1, sum_extend<typename PreviousState::extend, boost::mpl::_2> >
                >::type final_extend;

                typedef typename boost::mpl::push_front<typename PreviousState::list, _impl::wrap_type<raw_extends> >::type new_list;

                typedef state<new_list, final_extend> type;
            };

            typedef typename boost::mpl::reverse_fold<
                ListOfRanges,
                state<boost::mpl::vector0<>, extend<0,0,0,0,0,0> >,
                update_state<boost::mpl::_1, boost::mpl::_2>
            >::type final_state;

            typedef typename final_state::list type;
        };

        template <typename State, typename SubArray>
        struct keep_scanning {
            typedef typename boost::mpl::fold<
                typename SubArray::type,
                State,
                boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
            >::type type;
        };

        template <typename Array>
        struct linearize_extend_sizes {
            typedef typename boost::mpl::fold<Array,
                boost::mpl::vector0<>,
                boost::mpl::if_<
                    _impl::is_wrap_type<boost::mpl::_2>,
                    keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
                >
            >::type type;
        };

        template<typename MssDescriptor>
        struct mss_compute_extend_sizes
        {
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: invalid type");

            /**
             * \brief Here the extends are calculated recursively, in order for each functor's domain to embed all the domains of the functors he depends on.
             */
            typedef typename boost::mpl::fold<
                typename mss_descriptor_esf_sequence<MssDescriptor>::type,
                boost::mpl::vector0<>,
                traverse_extends<boost::mpl::_1,boost::mpl::_2>
            >::type extends_list;

            /*
             *  Compute prefix sum to compute bounding boxes for calling a given functor
             */
            typedef typename prefix_on_extends<extends_list>::type structured_extend_sizes;

            /**
             * linearize the data flow graph
             *
             */
            typedef typename linearize_extend_sizes<structured_extend_sizes>::type type;

            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size<typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type>::value ==
                 boost::mpl::size<type>::value), "Internal Error: wrong size");
        };

        template <typename Placeholders>
        struct compute_extends_of {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<Placeholders, is_arg>::value), "wrong type");
            template<typename MssDescriptor>
            struct for_mss
            {
                GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: invalid type");
    
                template <typename PLH>
                struct map_of_empty_extends {
                    typedef typename boost::mpl::fold<
                        PLH,
                        boost::mpl::map0<>,
                        boost::mpl::insert<boost::mpl::_1,
                                           boost::mpl::pair<boost::mpl::_2, extend<0,0,0,0,0,0> >
                                           >
                        >::type type;
                };
    
                template <typename CurrentRange>
                struct work_on {
                    template <typename PlcRangePair, typename CurrentMap>
                    struct with {
                        typedef typename sum_extend<CurrentRange, typename PlcRangePair::second>::type candidate_extend;
                        typedef typename enclosing_extend<candidate_extend, typename boost::mpl::at<CurrentMap, typename PlcRangePair::first>::type>::type extend;
                        typedef typename boost::mpl::erase_key<CurrentMap, typename PlcRangePair::first>::type map_erased;
                        typedef typename boost::mpl::insert<map_erased, boost::mpl::pair<typename PlcRangePair::first, extend> >::type type; // new map
                    };
                };
    
                template <typename ESFs, typename CurrentMap, int Elements>
                struct populate_map {
                    typedef typename boost::mpl::at_c<ESFs, 0>::type current_ESF;
                    typedef typename boost::mpl::pop_front<ESFs>::type rest_of_ESFs;
    
                    typedef typename esf_get_the_only_w_per_functor<current_ESF, boost::false_type>::type output;
                    // ^^^^ they (must) have the same extend<0,0,0,0,0,0> [so not need for true predicate]
                    // now assuming there is only one
    
                    typedef typename esf_get_r_per_functor<current_ESF, boost::true_type>::type inputs;
    
                    typedef typename boost::mpl::at<CurrentMap, output>::type current_extend;
    
                    typedef typename boost::mpl::fold<
                        inputs,
                        CurrentMap,
                        typename work_on<current_extend>::template with<boost::mpl::_2,boost::mpl::_1>
                        >::type new_map;
    
                    typedef typename populate_map<rest_of_ESFs, new_map, boost::mpl::size<rest_of_ESFs>::type::value >::type type;
                };
    
                template <typename ESFs, typename CurrentMap>
                struct populate_map<ESFs, CurrentMap, 0> {
                    typedef CurrentMap type;
                };
    
                typedef typename boost::mpl::reverse<typename mss_descriptor_esf_sequence<MssDescriptor>::type>::type ESFs;
    
                typedef typename populate_map<ESFs,
                                              typename map_of_empty_extends<Placeholders>::type,
                                              boost::mpl::size<ESFs>::type::value >::type type;
    
            };
        };
    }
}
