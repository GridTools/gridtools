#pragma once

#include "make_stencils.h"
#include <boost/mpl/transform.hpp>
#include <boost/mpl/for_each.hpp>
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

namespace gridtools {
    namespace _impl{
        /* wrap type to simplify specialization based on mpl::vectors */
        template <typename mpl_array>
        struct wrap_type {
            typedef mpl_array type;
        };

        template <typename T>
        struct is_wrap_type {
            typedef typename boost::false_type type;
        };

        template <typename T>
        struct is_wrap_type<wrap_type<T> > {
            typedef typename boost::true_type type;
        };



        /* Few short and abvious metafunctions */
        struct extract_functor {
            template <typename T>
            struct apply {
                typedef typename T::esf_function type;
            };
        };

        template <typename t_dom, template <class A, class B> class t_local_domain>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef t_local_domain<T, t_dom> type;
            };
        };
    
        /* Functor used to instantiate the local domains to be passed to each
           elementary stencil function */
        template <typename t_dom>
        struct instantiate_local_domain {
            t_dom * dom;
            instantiate_local_domain(t_dom * dom)
                : dom(dom)
            {}

            template <typename t_elem>
            void operator()(t_elem & elem) const {
                elem.init(dom, 0,0,0);
            }
        };

        template <typename t_functor_desc>
        struct extract_ranges {
            typedef typename t_functor_desc::esf_function t_functor;

            template <typename range_state, typename argument_index>
            struct update_range {
                typedef typename boost::mpl::at<typename t_functor::arg_list, argument_index>::type argument_type;
                typedef typename enclosing_range<range_state, typename argument_type::range_type>::type type;
            };

            typedef typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, t_functor::n_args>,
                range<0,0,0,0>,
                update_range<boost::mpl::_1, boost::mpl::_2>
                >::type type;
        };

        template <typename T>
        struct extract_ranges<independent_esf<T> >
        {
            typedef typename boost::false_type type;
        };

        template <typename not_independent_elem>
        struct from_independents {
            typedef typename boost::false_type type;
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

    template <typename state, typename elem>
    struct traverse_ranges {

        typedef typename boost::mpl::push_back<
            state,
            typename boost::mpl::if_<
                is_independent<elem>,
                typename from_independents<elem>::type,
                typename extract_ranges<elem>::type
                >::type
            >::type type;
};


        template <typename list_of_ranges>
        struct prefix_on_ranges {

            template <typename t_list, typename t_range/*, typename t_next_range*/>
            struct state {
                typedef t_list list;
                typedef t_range range;
                // typedef t_next_range next_range;
            };

            template <typename previous_state, typename current_element>
            struct update_state {
                typedef typename sum_range<typename previous_state::range,
                                           current_element>::type new_range;
                typedef typename boost::mpl::push_front<typename previous_state::list, typename previous_state::range>::type new_list;
                typedef state<new_list, new_range> type;
            };

            template <typename previous_state, typename ind_vector>
            struct update_state<previous_state, wrap_type<ind_vector> > {
                typedef typename boost::mpl::fold<
                    ind_vector,
                    boost::mpl::vector<>,
                    boost::mpl::push_back<boost::mpl::_1, /*sum_range<*/typename previous_state::range/*, boost::mpl::_2>*/ >
                    >::type raw_ranges;
            
            typedef typename boost::mpl::fold<
                ind_vector,
                range<0,0,0,0>,
                enclosing_range<boost::mpl::_1, sum_range<typename previous_state::range, boost::mpl::_2> >
                >::type final_range;

            typedef typename boost::mpl::push_front<typename previous_state::list, wrap_type<raw_ranges> >::type new_list;

            typedef state<new_list, final_range> type;
        };

        typedef typename boost::mpl::reverse_fold<
            list_of_ranges,
            state<boost::mpl::vector<>, range<0,0,0,0> >,
            update_state<boost::mpl::_1, boost::mpl::_2> >::type final_state;

            typedef typename final_state::list type;
        };

        template <typename state, typename subarray>
        struct keep_scanning {
            typedef typename boost::mpl::fold<
                typename subarray::type,
                state,
                boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
                >::type type;
        };

        template <typename array>
        struct linearize_range_sizes {
            typedef typename boost::mpl::fold<array,
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
        template <typename t_coords>
        struct show_pair {
            t_coords coords;

            explicit show_pair(t_coords const& coords)
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

            template <int i, int j, int k, int l>
            void operator()(range<i,j,k,l> const&) const {
                std::cout << prefix << range<i,j,k,l>() << std::endl;
            }

            template <typename mplvec>
            void operator()(mplvec const&) const {
                std::cout << "Independent" << std::endl;
                boost::mpl::for_each<mplvec>(print__(std::string("    ")));
                std::cout << "End Independent" << std::endl;
            }

            template <typename mplvec>
            void operator()(_impl::wrap_type<mplvec> const&) const {
                std::cout << "Independent" << std::endl;
                boost::mpl::for_each<mplvec>(print__(std::string("    ")));
                std::cout << "End Independent" << std::endl;
            }
        };

    } // namespace _debug

    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide coordinates to the functor
     * and one that does not
     * 
     * @tparam t_esf_descriptor The descriptor of the elementary stencil function
     * @tparam t_domain The full domain type
     */
    template <typename t_esf_descriptor, typename t_domain>
    struct local_domain_base {
        typedef typename t_esf_descriptor::args esf_args;
        typedef typename t_esf_descriptor::esf_function esf_function;
        typedef typename t_domain::placeholders dom_placeholders;
        typedef t_domain domain_type;

        t_domain *dom;

        int m_i,m_j,m_k;

                    
        void init(t_domain* _dom) {
            dom = _dom;
        }

        template <typename T>
        typename boost::mpl::at<esf_args, typename T::index_type>::type::value_type&  
        operator()(T const& t) const {
            return dom->template direct<typename boost::mpl::template at<esf_args, typename T::index_type>::type::index_type>(/*typename T::index()*/);
        }

        template <typename T>
        typename boost::mpl::at<esf_args, typename T::index>::type::value_type& 
        operator()(T const&, int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::template at<esf_args, typename T::index>::type::index>();
        }

        template <typename T>
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        get(int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::template at_c<esf_args, T::index>::type::index>();     
        }

        template <typename T>
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        operator[](T const&) const {
            return dom->template direct<boost::mpl::template at_c<esf_args, T::index>::type::index>();
        }

        void move_to(int i, int j, int k) const {
            dom->move_to(i,j,k);
        }

        void increment() const {
            dom->template increment_along<2>();
        }

    };

    //            template <typename t_esf_descriptor, typename t_domain>
    //            struct local_domain_location : public local_domain_base<t_esf_descriptor, t_domain> {
    //                typedef local_domain_base<t_esf_descriptor, t_domain> base_type;
    //                typedef typename t_esf_descriptor::args esf_args;
    //                typedef typename t_esf_descriptor::esf_function esf_function;
    //                typedef typename t_domain::placeholders dom_placeholders;
    //                //typedef typename t_domain::arg dom_args;
    //                typedef t_domain domain_type;
    //
    //                int m_i,m_j,m_k;
    //
    //                explicit local_domain_location(t_domain const & dom, int i, int j, int k)
    //                    : base_type(dom)
    //                    , m_i(i)
    //                    , m_j(j)
    //                    , m_k(k)
    //                {
    //                    std::cout << "LOCAL DOMAIN LOCATIONNNNN" << std::endl;
    //                }
    //
    //                int i() const { return m_i;}
    //                int j() const { return m_j;}
    //                int k() const { return m_k;}
    //            };

    /**
     * This class extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. This version does not provide coordinates
     * to the function operator
     * 
     * @tparam t_esf_descriptor The descriptor of the elementary stencil function
     * @tparam t_domain The full domain type
     */
    template <typename t_esf_descriptor, typename t_domain>
    struct local_domain : public local_domain_base<t_esf_descriptor, t_domain> {
        typedef local_domain_base<t_esf_descriptor, t_domain> base_type;
        typedef typename t_esf_descriptor::args esf_args;
        typedef typename t_esf_descriptor::esf_function esf_function;
        typedef typename t_domain::placeholders dom_placeholders;
        typedef t_domain domain_type;

        local_domain() {}
                
        void init(t_domain* dom, int, int, int)
        {
            base_type::init(dom);
#ifndef NDEBUG
            std::cout << "LOCAL DOMAIN" << std::endl;
#endif
        }

        int i() const {return; }
        int j() const {return; }
        int k() const {return; }
    };

    struct intermediate {

        template <typename t_backend, typename t_mss_type, typename t_domain_type, typename t_coords>
        static void run(t_mss_type const &, t_domain_type & domain, t_coords const & coords)
        {
            typedef typename boost::mpl::transform<typename t_mss_type::linear_esf,
                _impl::extract_functor>::type functors_list;
        
            //typedef typename extend_by<typename t_coords::axis_type, 1>::type ext_axis;
        
            // compute the functor do methods - This is the most computationally intensive part
            typedef typename boost::mpl::transform<
            functors_list,
                compute_functor_do_methods<boost::mpl::_, typename t_coords::axis_type>
                >::type FunctorDoMethods; // Vector of vectors - each element is a vector of pairs of actual axis-indices

            // compute the loop intervals
            typedef typename compute_loop_intervals<
                FunctorDoMethods,
                typename t_coords::axis_type
                >::type LoopIntervals; // vector of pairs of indices - sorted and contiguous

            // compute the do method lookup maps
            typedef typename boost::mpl::transform<
                FunctorDoMethods,
                compute_functor_do_method_lookup_map<boost::mpl::_, LoopIntervals>
                >::type FunctorDoMethodLookupMaps; // vector of maps, indexed by functors indices in Functor vector. 
            // Each map key is a pair of indices in the axis, value is the corresponding method interval.

#ifndef NDEBUG
            std::cout << "Actual loop bounds ";
            boost::mpl::for_each<LoopIntervals>(_debug::show_pair<t_coords>(coords));
            std::cout << std::endl;
#endif
        
            // Create a fusion::vector of domains for each functor
            typedef typename boost::mpl::transform<
                typename t_mss_type::linear_esf,
                typename _impl::get_local_domain<t_domain_type, local_domain> >::type mpl_local_domain_list;
            typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type t_local_domain_list;

            t_local_domain_list local_domain_list;
            boost::fusion::for_each(local_domain_list, 
                                    _impl::instantiate_local_domain<t_domain_type>
                                    (const_cast<typename boost::remove_const<t_domain_type>::type*>(&domain)));

            // Extract the ranges from functors to determine iteration spaces bounds

            // For each functor collect the minimum enclosing box of the ranges for the arguments
            typedef typename boost::mpl::fold<typename t_mss_type::esf_array,
                                              boost::mpl::vector<>,
                                              _impl::traverse_ranges<boost::mpl::_1,boost::mpl::_2>
                                              >::type ranges_list;


#ifndef NDEBUG
            std::cout << "ranges list" << std::endl;
            boost::mpl::for_each<ranges_list>(_debug::print__());
#endif
        
            // Compute prefix sum to compute bounding boxes for calling a given functor
            typedef typename _impl::prefix_on_ranges<ranges_list>::type structured_range_sizes;

#ifndef NDEBUG
            std::cout << "range sizes" << std::endl;
            boost::mpl::for_each<structured_range_sizes>(_debug::print__());
            std::cout << "end1" <<std::endl;
#endif
        
            typedef typename _impl::linearize_range_sizes<structured_range_sizes>::type range_sizes;

#ifndef NDEBUG
            boost::mpl::for_each<range_sizes>(_debug::print__());
            std::cout << "end2" <<std::endl;
#endif
        
            int tileI = (t_backend::BI)?
                (t_backend::BI):
                (coords.i_high_bound()-coords.i_low_bound()+1);
        
            int tileJ = (t_backend::BJ)?
                (t_backend::BJ):
                (coords.j_high_bound()-coords.j_low_bound()+1);


            /*******
                    The following couple of calls should be decoupled from <run>
                    since it may be costly to do that work everytime a stencil
                    is executed.
             *******/
            // Prepare domain's temporary fields to proper storage sizes
            domain.template prepare_temporaries<t_mss_type, range_sizes, t_backend>
                (tileI,
                 tileJ, 
                 coords.value_at_top()-coords.value_at_bottom()+1);

            domain.setup_computation();
            // Now run!
            t_backend::template run<functors_list, range_sizes, LoopIntervals, FunctorDoMethodLookupMaps>(domain, coords, local_domain_list);
            domain.finalize_computation();
        }    
    };

} // namespace gridtools
