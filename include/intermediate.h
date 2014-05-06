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

namespace gridtools {
    namespace _impl{
        /* wrap type to simplify specialization based on mpl::vectors */
        template <typename MplArray>
        struct wrap_type {
            typedef MplArray type;
        };

        template <typename T>
        struct is_wrap_type
          : boost::false_type
        {};

        template <typename T>
        struct is_wrap_type<wrap_type<T> >
          : boost::true_type
        {};



        /* Few short and abvious metafunctions */
        struct extract_functor {
            template <typename T>
            struct apply {
                typedef typename T::esf_function type;
            };
        };

        template <typename Dom, template <class A, class B> class LocalDomain>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef LocalDomain<T, Dom> type;
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



    template <typename Backend, typename MssType, typename DomainType, typename Coords>
    struct intermediate : public computation {

        // typename MssType::linear_esf is a list of all the esf nodes in the multi-stage descriptor.
        // functors_list is a list of all the functors of all the esf nodes in the multi-stage descriptor.
        typedef typename boost::mpl::transform<typename MssType::linear_esf,
                                               _impl::extract_functor>::type functors_list;
        
        // compute the functor do methods - This is the most computationally intensive part
        typedef typename boost::mpl::transform<
            functors_list,
            compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
            >::type FunctorDoMethods; // Vector of vectors - each element is a vector of pairs of actual axis-indices

        // compute the loop intervals
        typedef typename compute_loop_intervals<
            FunctorDoMethods,
            typename Coords::axis_type
            >::type LoopIntervals; // vector of pairs of indices - sorted and contiguous

        // compute the do method lookup maps
        typedef typename boost::mpl::transform<
                FunctorDoMethods,
                compute_functor_do_method_lookup_map<boost::mpl::_, LoopIntervals>
                >::type FunctorDoMethodLookupMaps; // vector of maps, indexed by functors indices in Functor vector. 


        // Create a fusion::vector of domains for each functor
        typedef typename boost::mpl::transform<
            typename MssType::linear_esf,
            _impl::get_local_domain<DomainType, local_domain> >::type mpl_local_domain_list;

        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;

        LocalDomainList local_domain_list;

        typedef typename boost::mpl::fold<typename MssType::esf_array,
                                          boost::mpl::vector<>,
                                          _impl::traverse_ranges<boost::mpl::_1,boost::mpl::_2>
                                          >::type ranges_list;

        // Compute prefix sum to compute bounding boxes for calling a given functor
        typedef typename _impl::prefix_on_ranges<ranges_list>::type structured_range_sizes;

        // linearize the data flow graph.
        typedef typename _impl::linearize_range_sizes<structured_range_sizes>::type range_sizes;

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

            tileI = (Backend::BI)?
                (Backend::BI):
                (coords.i_high_bound()-coords.i_low_bound()+1);
        
            tileJ = (Backend::BJ)?
                (Backend::BJ):
                (coords.j_high_bound()-coords.j_low_bound()+1);

#ifndef NDEBUG
            std::cout << "tileI " << tileI << " "
                      << "tileK " << tileJ
                      << std::endl;
#endif

            /*******
                    The following couple of calls should be decoupled from <run>
                    since it may be costly to do that work everytime a stencil
                    is executed.
             *******/
            // Prepare domain's temporary fields to proper storage sizes
            // domain.template prepare_temporaries<MssType, range_sizes>
            //     (tileI,
            //      tileJ, 
            //      coords.value_at_top()-coords.value_at_bottom()+1);

            // domain.setup_computation();
            // // Now run!
            // Backend::template run<functors_list, range_sizes, LoopIntervals, FunctorDoMethodLookupMaps>(domain, coords, local_domain_list);
            // domain.finalize_computation();
        }    

        void ready () {
            m_domain.template prepare_temporaries<MssType, range_sizes>
                (tileI,
                 tileJ, 
                 m_coords.value_at_top()-m_coords.value_at_bottom());
        }

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
            m_domain.finalize_computation();
        }

        void run () {
            Backend::template run<functors_list, range_sizes, LoopIntervals, FunctorDoMethodLookupMaps>(m_domain, m_coords, local_domain_list);
        }

    private:
        int tileI, tileJ;


    };

} // namespace gridtools
