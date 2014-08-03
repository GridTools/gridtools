#pragma once

#include "make_stencils.h"
#include <boost/mpl/transform.hpp>
#include "gt_for_each/for_each.hpp"
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/type_traits/remove_const.hpp>
#include "level.h"
#include "interval.h"
#include "loopintervals.h"
#include "functor_do_methods.h"
#include "functor_do_method_lookup_maps.h"
#include "axis.h"
#include "local_domain.h"
#include "computation.h"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */

namespace gridtools {
    namespace _impl{
        /**@brief wrap type to simplify specialization based on mpl::vectors */
        template <typename MplArray>
        struct wrap_type {
            typedef MplArray type;
        };

        /**
         * @brief compile-time boolean operator returning true if the template argument is a wrap_type
         * */
        template <typename T>
        struct is_wrap_type
          : boost::false_type
        {};

        template <typename T>
        struct is_wrap_type<wrap_type<T> >
          : boost::true_type
        {};


        template <typename Backend, typename RangeSizes>
        struct obtain_type_from_backend {
            template <typename ElemType>
            struct apply {
                typedef ElemType type;
            };

            template <typename StorageType>
            struct apply<no_storage_type_yet<StorageType> > {
                typedef host_tmp_storage< typename StorageType::value_type, typename StorageType::layout_type> type;;
            };
        };

        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        struct extract_functor {
            template <typename T>
            struct apply {
                typedef typename T::esf_function type;
            };
        };


        template <typename StoragePointers, typename Iterators, template <class A, class B, class C> class LocalDomain>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef LocalDomain<StoragePointers,Iterators,T> type;
            };
        };

        /* Functor used to instantiate the local domains to be passed to each
           elementary stencil function */
        template <typename Dom>
        struct instantiate_local_domain {
            Dom * dom;
            GT_FUNCTION
            instantiate_local_domain(Dom * dom)
                : dom(dom)
            {}

            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                elem.init(dom, 0,0,0);
                elem.clone_to_gpu();
            }
        };

        template <typename FunctorDesc>
        struct extract_ranges {
            typedef typename FunctorDesc::esf_function Functor;

            template <typename RangeState, typename ArgumentIndex>
            struct update_range {
                typedef typename boost::mpl::at<typename Functor::arg_list, ArgumentIndex>::type argument_type;
                typedef typename enclosing_range<RangeState, typename argument_type::range_type>::type type;
            };

            typedef typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, Functor::n_args>,
                range<0,0,0,0>,
                update_range<boost::mpl::_1, boost::mpl::_2>
                >::type type;
        };

        template <typename T>
        struct extract_ranges<independent_esf<T> >
        {
            typedef boost::false_type type;
        };

        template <typename NotIndependentElem>
        struct from_independents {
            typedef boost::false_type type;
        };

        template <typename T>
        struct from_independents<independent_esf<T> > {
            typedef typename boost::mpl::fold<
                typename independent_esf<T>::esf_list,
                boost::mpl::vector<>,
                boost::mpl::push_back<boost::mpl::_1, extract_ranges<boost::mpl::_2> >
                >::type raw_type;

        typedef wrap_type<raw_type> type;
    };

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


    // prefix sum, scan operation, takes into account the range needed by the current stage plus the range needed by the next stage.
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
            struct update_state<PreviousState, wrap_type<IndVector> > {
                typedef typename boost::mpl::fold<
                    IndVector,
                    boost::mpl::vector<>,
                    boost::mpl::push_back<boost::mpl::_1, /*sum_range<*/typename PreviousState::range/*, boost::mpl::_2>*/ >
                    >::type raw_ranges;

                typedef typename boost::mpl::fold<
                        IndVector,
                        range<0,0,0,0>,
                        enclosing_range<boost::mpl::_1, sum_range<typename PreviousState::range, boost::mpl::_2> >
                >::type final_range;

                typedef typename boost::mpl::push_front<typename PreviousState::list, wrap_type<raw_ranges> >::type new_list;

                typedef state<new_list, final_range> type;
            };

            typedef typename boost::mpl::reverse_fold<
                ListOfRanges,
                state<boost::mpl::vector<>, range<0,0,0,0> >,
                update_state<boost::mpl::_1, boost::mpl::_2> >::type final_state;

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
                                              boost::mpl::vector<>,
                                              boost::mpl::if_<
                                                  is_wrap_type<boost::mpl::_2>,
                                                  keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                                                  boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
                                                  >
            >::type type;
        };

/**
 * @}
 * */

    } //namespace _impl


    namespace _debug {
        template <typename Coords>
        struct show_pair {
            Coords coords;

            explicit show_pair(Coords const& coords)
                : coords(coords)
            {}

            template <typename T>
            void operator()(T const&) const {
                typedef typename index_to_level<typename T::first>::type from;
                typedef typename index_to_level<typename T::second>::type to;
                std::cout << "{ (" << from() << " "
                          << to() << ") "
                          << "[" << coords.template value_at<from>() << ", "
                          << coords.template value_at<to>() << "] } ";
            }
        };

        struct print__ {
            std::string prefix;

            print__()
                : prefix("")
            {}

            print__(std::string const &s)
                : prefix(s)
            {}

            template <int I, int J, int K, int L>
            void operator()(range<I,J,K,L> const&) const {
                std::cout << prefix << range<I,J,K,L>() << std::endl;
            }

            template <typename MplVector>
            void operator()(MplVector const&) const {
                std::cout << "Independent" << std::endl;
                gridtools::for_each<MplVector>(print__(std::string("    ")));
                std::cout << "End Independent" << std::endl;
            }

            template <typename MplVector>
            void operator()(_impl::wrap_type<MplVector> const&) const {
                printf("Independent*\n"); // this whould not be necessary but nvcc s#$ks
                gridtools::for_each<MplVector>(print__(std::string("    ")));
                printf("End Independent*\n");
            }
        };

    } // namespace _debug


/**
 * @class
 * @brief structure collecting helper metafunctions
 * */
    template <typename Backend, typename MssType, typename DomainType, typename Coords>
    struct intermediate : public computation {


        /**
         * typename MssType::linear_esf is a list of all the esf nodes in the multi-stage descriptor.
         * functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
         */
        typedef typename boost::mpl::transform<typename MssType::linear_esf,
                                               _impl::extract_functor>::type functors_list;
        
        /**
         *  compute the functor do methods - This is the most computationally intensive part
         */
        typedef typename boost::mpl::transform<
            functors_list,
            compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
            >::type functor_do_methods; // Vector of vectors - each element is a vector of pairs of actual axis-indices

        /**
         * compute the loop intervals
         */
        typedef typename compute_loop_intervals<
            functor_do_methods,
            typename Coords::axis_type
            >::type LoopIntervals; // vector of pairs of indices - sorted and contiguous

        /**
         * compute the do method lookup maps
         *
         */
        typedef typename boost::mpl::transform<
                functor_do_methods,
                compute_functor_do_method_lookup_map<boost::mpl::_, LoopIntervals>
                >::type functor_do_method_lookup_maps; // vector of maps, indexed by functors indices in Functor vector.


        /**
         *
         */
        typedef typename boost::mpl::fold<typename MssType::esf_array,
                                          boost::mpl::vector<>,
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
        typedef typename _impl::linearize_range_sizes<structured_range_sizes>::type range_sizes;

        /**
         * Takes the domain list of storage pointer types and transform
         * the no_storage_type_yet with the types provided by the
         * backend with the interface that takes the range sizes. This
         * must be done before getting the local_domain
         */
        typedef typename boost::mpl::transform<typename DomainType::arg_list, typename _impl::obtain_type_from_backend<Backend, range_sizes> >::type actual_arg_list;

        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform<
            typename MssType::linear_esf,
            _impl::get_local_domain<actual_arg_list, typename DomainType::iterator_list, local_domain> >::type mpl_local_domain_list;

        /**
         *
         */
        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;

        /**
         *
         */
        LocalDomainList local_domain_list;


        DomainType & m_domain;
        Coords m_coords;

        intermediate(MssType const &, DomainType & domain, Coords const & coords)
            : m_domain(domain)
            , m_coords(coords)
        {
            // Each map key is a pair of indices in the axis, value is the corresponding method interval.

#ifndef NDEBUG
#ifndef __CUDACC__
            std::cout << "Actual loop bounds ";
            gridtools::for_each<LoopIntervals>(_debug::show_pair<Coords>(coords));
            std::cout << std::endl;
#endif
#endif

            // Extract the ranges from functors to determine iteration spaces bounds

            // For each functor collect the minimum enclosing box of the ranges for the arguments

#ifndef NDEBUG
            std::cout << "ranges list" << std::endl;
            gridtools::for_each<ranges_list>(_debug::print__());
#endif

#ifndef NDEBUG
            std::cout << "range sizes" << std::endl;
            gridtools::for_each<structured_range_sizes>(_debug::print__());
            std::cout << "end1" <<std::endl;
#endif

#ifndef NDEBUG
            gridtools::for_each<range_sizes>(_debug::print__());
            std::cout << "end2" <<std::endl;
#endif
        }    
        /**
           @brief This method allocates on the heap the temporary variables.
           Calls heap_allocated_temps::prepare_temporaries(...).
           It allocates the memory for the list of ranges defined in the temporary placeholders.
         */
        void ready () {
            Backend::template prepare_temporaries<MssType, range_sizes>(m_domain, m_coords);
        }
        /**
           @brief calls setup_computation and creates the local domains.
           The constructors of the local domains get called
           (\ref gridtools::intermediate::instantiate_local_domain, which only initializes the dom public pointer variable)
           @note the local domains are allocated in the public scope of the \ref gridtools::intermediate struct, only the pointer
           is passed to the instantiate_local_domain struct
         */
        void steady () {
            m_domain.setup_computation();

            boost::fusion::for_each(local_domain_list,
                                    _impl::instantiate_local_domain<DomainType>
                                    (const_cast<typename boost::remove_const<DomainType>::type*>(&m_domain)));

#ifndef NDEBUG
            m_domain.info();
#endif

        }

        void finalize () {
            Backend::finalize_computation(m_domain);
        }

        /**
         * \brief the execution of the stencil operations take place in this call
         *
         */
        void run () {
            Backend::template run<functors_list, range_sizes, LoopIntervals, functor_do_method_lookup_maps>(m_domain, m_coords, local_domain_list);
        }

    };

} // namespace gridtools
