#pragma once

#include "make_stencils.h"
#include <boost/mpl/transform.hpp>
#include "gt_for_each/for_each.hpp"
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/type_traits/remove_const.hpp>
#include "level.h"
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
        template <typename Dom, typename ArgList>
        struct instantiate_local_domain {
            GT_FUNCTION
            instantiate_local_domain(Dom const& dom, ArgList const& arg_list)
                : m_dom(dom)
                , m_arg_list(arg_list)
            {}

            /**method called in the Do methods of the functors. Elem is a local_domain*/
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                elem.init(m_dom, m_arg_list, 0,0,0);
                elem.clone_to_gpu();
            }

        private:
            Dom const& m_dom;
            ArgList const& m_arg_list;
        };

        template <typename FunctorDesc>
        struct extract_ranges {
            typedef typename FunctorDesc::esf_function Functor;

            /**here the ranges for the functors are calculated*/
            template <typename RangeState, typename ArgumentIndex>
            struct update_range {
                typedef typename boost::mpl::at<typename Functor::arg_list, ArgumentIndex>::type argument_type;
                typedef typename enclosing_range<RangeState, typename argument_type::range_type>::type type;
            };

            /**here the ranges for the functors are calculated*/
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


        template <typename Index>
        struct has_index_ {
            typedef boost::mpl::int_<Index::value> val1;
            template <typename Elem>
            struct apply {
                typedef typename boost::mpl::int_<Elem::second::value> val2;

                typedef typename boost::is_same<val1, val2>::type type;
            };
        };

        template <typename Placeholders,
                  typename TmpPairs>
        struct select_storage {
            template <typename T, typename Dummy = void>
            struct is_temp : public boost::false_type
            { };

            template <typename T>
            struct is_temp<no_storage_type_yet<T> > : public boost::true_type
            { };

            template <bool b, typename Storage, typename tmppairs, typename index>
            struct get_the_type;

            template <typename Storage, typename tmppairs, typename index>
            struct get_the_type<true, Storage, tmppairs,index> {
                typedef typename boost::mpl::deref<
                    typename boost::mpl::find_if<
                        tmppairs,
                        has_index_<index>
                        >::type
                    >::type::first type;
            };

            template <typename Storage, typename tmppairs, typename index>
            struct get_the_type<false, Storage, tmppairs,index> {
                typedef Storage type;
            };

            template <typename Index>
            struct apply {
                typedef typename boost::mpl::at<Placeholders, Index>::type::storage_type storage_type;
                static const bool b = is_temp<storage_type>::value;
                typedef typename get_the_type<b, storage_type, TmpPairs, Index>::type* type;
            };
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


        struct _print_the_storages {
            template <typename T>
            void operator()(T const& x) const {
                //                int a =x;
                std::cout << "      AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  " << x << std::endl;
            }
        };
    } // namespace _debug


    struct _print_____ {
        template <typename T>
        void operator()(T) const {
            //int x = T();
            std::cout << "ciccia e brufoli " << T() << std::endl;
        }
    };

    struct _print______ {
        template <typename T>
        void operator()(T) const {
            std::cout << "   ==== == == ===  = == == = " << std::endl;
            gridtools::for_each<T>(_print_____());
        }
    };

    struct printthose {
        template <typename E>
        void operator()(E * e) const {
            std::cout << typename boost::remove_pointer<typename boost::remove_reference<E>::type>::type() << " std::hex " << std::hex << e << std::dec << "   " ;
        }
    };



//\todo move inside the traits classes
        template<enumtype::backend>
        struct finalize_computation;

        template<>
        struct finalize_computation<enumtype::Cuda>{
            template <typename DomainType>
            static void apply(DomainType& dom)
                {dom.finalize_computation();}
        };


        template<>
        struct finalize_computation<enumtype::Host>{
            template <typename DomainType>
            static void apply(DomainType& dom)
                {}
        };



//\todo move inside the traits classes?

        /**
           This functor calls h2d_update on all storages, in order to
           get the data prepared in the case of GPU execution.

           Returns 0 (GT_NO_ERRORS) on success
        */
        template<enumtype::backend>
        struct setup_computation;

        template<>
        struct setup_computation<enumtype::Cuda>{
            template<typename ArgListType, typename DomainType>
            static int apply(ArgListType& storage_pointers, DomainType &  domain){
                boost::fusion::copy(storage_pointers, domain.original_pointers);

                boost::fusion::for_each(storage_pointers, _impl::update_pointer());
#ifndef NDEBUG
                printf("POINTERS\n");
                boost::fusion::for_each(storage_pointers, _debug::print_pointer());
                printf("ORIGINAL\n");
                boost::fusion::for_each(domain.original_pointers, _debug::print_pointer());
#endif
                return GT_NO_ERRORS;
        }
        };

        template<>
        struct setup_computation<enumtype::Host>{
            template<typename ArgListType, typename DomainType>
            static int apply(ArgListType const& storage_pointers, DomainType &  domain){
                return GT_NO_ERRORS;
        }
        };


/**
 * @class
*  @brief structure collecting helper metafunctions
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
            >::type loop_intervals_t; // vector of pairs of indices - sorted and contiguous

        /**
         * compute the do method lookup maps
         *
         */
        typedef typename boost::mpl::transform<
                functor_do_methods,
                compute_functor_do_method_lookup_map<boost::mpl::_, loop_intervals_t>
                >::type functor_do_method_lookup_maps; // vector of maps, indexed by functors indices in Functor vector.


        /**
         * \brief Here the ranges are calculated recursively, in order for each functor's domain to embed all the domains of the functors he depends on.
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
        typedef typename Backend::template obtain_storage_types<DomainType, MssType, range_sizes>::type mpl_actual_tmp_pairs;

        typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename DomainType::placeholders>::type::value> iter_range;

        typedef typename boost::mpl::fold<
            iter_range,
            boost::mpl::vector<>,
            typename boost::mpl::push_back<
                boost::mpl::_1,
                typename _impl::select_storage<
                    typename DomainType::placeholders,
                    mpl_actual_tmp_pairs
                    >::template apply<boost::mpl::_2>
                >
            >::type mpl_actual_arg_list;

        //typedef typename Backend::template obtain_storage_types<DomainType, MssType, range_sizes>::written_temps_per_functor temp_list;

        //typedef typename Backend::template obtain_storage_types<DomainType, MssType, range_sizes>::temporaries tomp_list;

        typedef typename boost::fusion::result_of::as_vector<mpl_actual_arg_list>::type actual_arg_list_type;

        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform<
            typename MssType::linear_esf,
            _impl::get_local_domain<actual_arg_list_type, typename DomainType::iterator_list, local_domain> >::type mpl_local_domain_list;

        /**
         *
         */
        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;

        /**
         *
         */
        LocalDomainList local_domain_list;


        DomainType & m_domain;
        const Coords& m_coords;

        actual_arg_list_type actual_arg_list;

        struct printtypes {
            template <typename T>
            void operator()(T *) const {
                std::cout << T() << "            ----------" << std::endl;
            }
        };

        intermediate(MssType const &, DomainType & domain, Coords const & coords)
            : m_domain(domain)
            , m_coords(coords)
        {
            // Each map key is a pair of indices in the axis, value is the corresponding method interval.

#ifndef NDEBUG
#ifndef __CUDACC__
            std::cout << "Actual loop bounds ";
            gridtools::for_each<loop_intervals_t>(_debug::show_pair<Coords>(coords));
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

            typedef boost::fusion::filter_view<typename DomainType::arg_list,
                is_storage<boost::mpl::_1> > t_domain_view;

            typedef boost::fusion::filter_view<actual_arg_list_type,
                is_storage<boost::mpl::_1> > t_args_view;

            t_domain_view domain_view(domain.storage_pointers);
            t_args_view args_view(actual_arg_list);

            // boost::fusion::for_each(domain_view, printtypes());
            // boost::fusion::for_each(args_view, printtypes());

            boost::fusion::copy(domain_view, args_view);

            // std::cout << "\n(1) These are the view values" << std::endl;
            // boost::fusion::for_each(domain.storage_pointers, _debug::print_pointer());
            // std::cout << "\n(2) These are the view values" << std::endl;
            // boost::fusion::for_each(actual_arg_list, _debug::print_pointer());
        }
        /**
           @brief This method allocates on the heap the temporary variables.
           Calls heap_allocated_temps::prepare_temporaries(...).
           It allocates the memory for the list of ranges defined in the temporary placeholders.
         */
        void ready () {
            // boost::fusion::for_each(actual_arg_list, printthose());
            Backend::template prepare_temporaries( actual_arg_list, m_coords);
            is_storage_ready=true;
            // std::cout << "\n(3) These are the view values" << std::endl;
            // boost::fusion::for_each(actual_arg_list, _debug::print_pointer());
        }
        /**
           @brief calls setup_computation and creates the local domains.
           The constructors of the local domains get called
           (\ref gridtools::intermediate::instantiate_local_domain, which only initializes the dom public pointer variable)
           @note the local domains are allocated in the public scope of the \ref gridtools::intermediate struct, only the pointer
           is passed to the instantiate_local_domain struct
         */
        void steady () {
            if(is_storage_ready)
            {
                setup_computation<Backend::s_backend_id>::apply( actual_arg_list, m_domain );
#ifndef NDEBUG
                printf("Setup computation\n");
#endif
            }
            else
            {
#ifndef NDEBUG
                printf("Setup computation FAILED\n");
#endif
                exit( GT_ERROR_NO_TEMPS );
            }

            boost::fusion::for_each(local_domain_list,
                                    _impl::instantiate_local_domain<DomainType, actual_arg_list_type>
                                    (m_domain, actual_arg_list));

#ifndef NDEBUG
            m_domain.info();
#endif

        }



        void finalize () {
            finalize_computation<Backend::s_backend_id>::apply(m_domain);
        }


        /**
         * \brief the execution of the stencil operations take place in this call
         *
         */
        void run () {
            // std::cout <<"WAHTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT "
            //           << boost::mpl::size<mpl_actual_arg_list>::type::value
            //           << std::endl;

            // gridtools::for_each<typename DomainType::placeholders>(_print_____());
            // gridtools::for_each<temp_list>(_print______());
            // std::cout << "---" << std::endl;
            // gridtools::for_each<tomp_list>(_print_____());
            // std::cout << "--- ---" << std::endl;
            // gridtools::for_each<mpl_local_domain_list>(_print_____());

            boost::fusion::for_each(actual_arg_list, _debug::_print_the_storages());
            Backend::template run<functors_list, range_sizes, loop_intervals_t, functor_do_method_lookup_maps, typename MssType::execution_engine_t>( m_coords, local_domain_list );
        }

    private:
        bool is_storage_ready;
    };

} // namespace gridtools
