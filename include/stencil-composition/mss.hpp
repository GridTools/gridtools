#pragma once
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/map/map0.hpp>
#include <boost/mpl/assert.hpp>
#include "functor_do_methods.hpp"
#include "esf.hpp"
#include "../common/meta_array.hpp"
#include "caches/cache_metafunctions.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/mpl/erase_key.hpp>

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    namespace _impl
    {
        /**@brief wrap type to simplify specialization based on mpl::vectors */
        template <typename MplArray>
        struct wrap_type {
            typedef MplArray type;
        };

        /**
         * @brief compile-time boolean operator returning true if the template argument is a wrap_type
         * */
        template <typename T>
        struct is_wrap_type : boost::false_type {};

        template <typename T>
        struct is_wrap_type<wrap_type<T> > : boost::true_type{};


        struct extract_functor {
            template <typename T>
            struct apply {
                typedef typename T::esf_function type;
            };
        };

        template <typename FunctorDesc>
        struct extract_ranges {
            typedef typename FunctorDesc::esf_function Functor;

            /**@brief here the ranges for the functors are calculated: the resulting type will be the range (i,j) which is enclosing all the ranges of the field used by the specific functor*/
            template <typename RangeState, typename ArgumentIndex>
            struct update_range {
                typedef typename boost::mpl::at<typename Functor::arg_list, ArgumentIndex>::type argument_type;
                typedef typename enclosing_range<RangeState, typename argument_type::range_type>::type type;
            };

            /**@brief here the ranges for the functors are calculated: iterates over the fields and calls the metafunction above*/
            typedef typename boost::mpl::fold<
                boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename Functor::arg_list>::type::value >,
                range<0,0,0,0,0,0>,
                update_range<boost::mpl::_1, boost::mpl::_2>
                >::type type;
        };

        template <typename NotIndependentElem>
        struct from_independents {
            typedef boost::false_type type;
        };

        /**@brief specialization for "independent" elementary stencil functions: given the list of  functors inside an elementary stencil function (esf) returns a vector of enclosing ranges, one per functor*/
        template <typename T>
        struct from_independents<independent_esf<T> > {
            typedef typename boost::mpl::fold<
                typename independent_esf<T>::esf_list,
                boost::mpl::vector0<>,
                boost::mpl::push_back<boost::mpl::_1, extract_ranges<boost::mpl::_2> >
            >::type raw_type;

            typedef wrap_type<raw_type> type;
        };

        template <typename T>
        struct extract_ranges<independent_esf<T> >
        {
            typedef boost::false_type type;
        };


        /** @brief metafunction returning, given the elementary stencil function "Elem", either the vector of enclosing ranges (in case of "independent" esf), or the single range enclosing all the ranges. */
        template <typename State, typename Elem>
            struct traverse_ranges {
                typedef typename boost::mpl::push_back<
                    State,
                    typename boost::mpl::if_<
                        is_independent<Elem>,
                        typename from_independents<Elem>::type,
                        typename extract_ranges<Elem>::type
                    >::type
                >::type type;
        };

        /**@brief prefix sum, scan operation, takes into account the range needed by the current stage plus the range needed by the next stage.*/
        template <typename ListOfRanges>
        struct prefix_on_ranges {

            template <typename List, typename Range/*, typename NextRange*/>
            struct state {
                typedef List list;
                typedef Range range;
                // typedef NextRange next_range;
            };

            template <typename PreviousState, typename CurrentElement>
            struct update_state {
                typedef typename sum_range<typename PreviousState::range,
                                               CurrentElement>::type new_range;
                typedef typename boost::mpl::push_front<typename PreviousState::list, typename PreviousState::range>::type new_list;
                typedef state<new_list, new_range> type;
            };

            template <typename PreviousState, typename IndVector>
            struct update_state<PreviousState, wrap_type<IndVector> >
            {
                typedef typename boost::mpl::fold<
                    IndVector,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back<boost::mpl::_1, /*sum_range<*/typename PreviousState::range/*, boost::mpl::_2>*/ >
                >::type raw_ranges;

                typedef typename boost::mpl::fold<
                    IndVector,
                    range<0,0,0,0,0,0>,
                    enclosing_range<boost::mpl::_1, sum_range<typename PreviousState::range, boost::mpl::_2> >
                >::type final_range;

                typedef typename boost::mpl::push_front<typename PreviousState::list, wrap_type<raw_ranges> >::type new_list;

                typedef state<new_list, final_range> type;
            };

            typedef typename boost::mpl::reverse_fold<
                ListOfRanges,
                state<boost::mpl::vector0<>, range<0,0,0,0,0,0> >,
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
        struct linearize_range_sizes {
            typedef typename boost::mpl::fold<Array,
                boost::mpl::vector0<>,
                boost::mpl::if_<
                    is_wrap_type<boost::mpl::_2>,
                    keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
                >
            >::type type;
        };


    }

    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence = boost::mpl::vector0<> >
    struct mss_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfDescrSequence, is_esf_descriptor>::value), "Internal Error: invalid type");

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<CacheSequence, is_cache>::value),
                "Internal Error: invalid type");
        typedef EsfDescrSequence esf_sequence_t;
        typedef CacheSequence cache_sequence_t;
    };

    template<typename mss>
    struct is_mss_descriptor : boost::mpl::false_{};

    template <typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence>
    struct is_mss_descriptor<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> > : boost::mpl::true_{};

    template<typename Mss>
    struct mss_descriptor_esf_sequence {};

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_esf_sequence<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        typedef EsfDescrSequence type;
    };

    template<typename T>
    struct mss_descriptor_linear_esf_sequence;

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_linear_esf_sequence<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        template <typename State, typename SubArray>
        struct keep_scanning
          : boost::mpl::fold<
                typename SubArray::esf_list,
                State,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
            >
        {};

        template <typename Array>
        struct linearize_esf_array : boost::mpl::fold<
              Array,
              boost::mpl::vector<>,
              boost::mpl::if_<
                  is_independent<boost::mpl::_2>,
                  keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                  boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
              >
        >{};

        typedef typename linearize_esf_array<EsfDescrSequence>::type type;
    };

    template<typename Mss>
    struct mss_descriptor_execution_engine {};

    template <typename ExecutionEngine,
              typename EsfDescrSequence,
              typename CacheSequence>
    struct mss_descriptor_execution_engine<mss_descriptor<ExecutionEngine, EsfDescrSequence, CacheSequence> >
    {
        typedef ExecutionEngine type;
    };

    template<typename MssDescriptor>
    struct mss_compute_range_sizes
    {
        GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: invalid type");

        /**
         * \brief Here the ranges are calculated recursively, in order for each functor's domain to embed all the domains of the functors he depends on.
         */
        typedef typename boost::mpl::fold<
            typename mss_descriptor_esf_sequence<MssDescriptor>::type,
            boost::mpl::vector0<>,
            _impl::traverse_ranges<boost::mpl::_1,boost::mpl::_2>
        >::type ranges_list;

        /*
         *  Compute prefix sum to compute bounding boxes for calling a given functor
         */
        typedef typename _impl::prefix_on_ranges<ranges_list>::type structured_range_sizes;

        /**
         * linearize the data flow graph
         *
         */
        typedef typename _impl::linearize_range_sizes<structured_range_sizes>::type type;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size<typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type>::value ==
             boost::mpl::size<type>::value), "Internal Error: wrong size");
    };


    template <typename Placeholders>
    struct compute_ranges_of {
        template<typename MssDescriptor>
        struct for_mss
        {
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor<MssDescriptor>::value), "Internal Error: invalid type");

            template <typename PLH>
            struct map_of_empty_ranges {
                typedef typename boost::mpl::fold<
                    PLH,
                    boost::mpl::map0<>,
                    boost::mpl::insert<boost::mpl::_1,
                                       boost::mpl::pair<boost::mpl::_2, range<0,0,0,0,0,0> >
                                       >
                    >::type type;
            };

            template <typename CurrentRange>
            struct work_on {
                template <typename PlcRangePair, typename CurrentMap>
                struct with {
                    typedef typename sum_range<CurrentRange, typename PlcRangePair::second>::type candidate_range;
                    typedef typename enclosing_range<candidate_range, typename boost::mpl::at<CurrentMap, typename PlcRangePair::first>::type>::type range;
                    typedef typename boost::mpl::erase_key<CurrentMap, typename PlcRangePair::first>::type map_erased;
                    typedef typename boost::mpl::insert<map_erased, boost::mpl::pair<typename PlcRangePair::first, range> >::type type; // new map
                };
            };

            template <typename ESFs, typename CurrentMap, int Elements>
            struct populate_map {
                typedef typename boost::mpl::at_c<ESFs, 0>::type current_ESF;
                typedef typename boost::mpl::pop_front<ESFs>::type rest_of_ESFs;

                typedef typename esf_get_the_only_w_per_functor<current_ESF, boost::false_type>::type output;
                // ^^^^ they (must) have the same range<0,0,0,0,0,0> [so not need for true predicate]
                // now assuming there is only one

                typedef typename esf_get_r_per_functor<current_ESF, boost::true_type>::type inputs;

                typedef typename boost::mpl::at<CurrentMap, output>::type current_range;

                typedef typename boost::mpl::fold<
                    inputs,
                    CurrentMap,
                    typename work_on<current_range>::template with<boost::mpl::_2,boost::mpl::_1>
                    >::type new_map;

                typedef typename populate_map<rest_of_ESFs, new_map, boost::mpl::size<rest_of_ESFs>::type::value >::type type;
            };

            template <typename ESFs, typename CurrentMap>
            struct populate_map<ESFs, CurrentMap, 0> {
                typedef CurrentMap type;
            };

            typedef typename boost::mpl::reverse<typename mss_descriptor_esf_sequence<MssDescriptor>::type>::type ESFs;

            typedef typename populate_map<ESFs,
                                          typename map_of_empty_ranges<Placeholders>::type,
                                          boost::mpl::size<ESFs>::type::value >::type type;

        };
    };

} // namespace gridtools
