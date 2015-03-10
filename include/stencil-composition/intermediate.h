#pragma once
#include "make_stencils.h"
#include <boost/mpl/transform.hpp>
#include "gt_for_each/for_each.hpp"
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/copy.hpp>
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
#include "heap_allocated_temps.h"
#include "mss_local_domain.h"
#include "common/meta_array.h"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */

namespace gridtools {
    namespace _impl{


//        /*
//         *
//         * @name Few short and obvious metafunctions
//         * @{
//         * */
//        template <typename StoragePointers, typename Iterators, template <class A, class B, class C> class LocalDomain>
//        struct get_local_domain {
//            template <typename T>
//            struct apply {
//                typedef LocalDomain<StoragePointers,Iterators,T> type;
//            };
//        };

    /** @brief Functor used to instantiate the local domains to be passed to each
        elementary stencil function */
    template <typename Dom, typename ArgList>
    struct instantiate_local_domain {
        GT_FUNCTION
        instantiate_local_domain(Dom const& dom, ArgList const& arg_list)
            : m_dom(dom)
            , m_arg_list(arg_list)
        {}

        /**Elem is a local_domain*/
        template <typename Elem>
        GT_FUNCTION
        void operator()(Elem & elem) const {
            BOOST_STATIC_ASSERT((is_local_domain<Elem>::value));

            elem.init(m_dom, m_arg_list, 0,0,0);
            elem.clone_to_gpu();
        }

    private:
        Dom const& m_dom;
        ArgList const& m_arg_list;
    };

    /** @brief Functor used to instantiate the local domains to be passed to each
        elementary stencil function */
    template <typename Dom, typename ArgList>
    struct instantiate_mss_local_domain {
        GT_FUNCTION
        instantiate_mss_local_domain(Dom const& dom, ArgList const& arg_list)
            : m_dom(dom)
            , m_arg_list(arg_list)
        {}

        /**Elem is a local_domain*/
        template <typename Elem>
        GT_FUNCTION
        void operator()(Elem & mss_local_domain_list) const {
            BOOST_STATIC_ASSERT((is_mss_local_domain<Elem>::value));

            boost::fusion::for_each(mss_local_domain_list.local_domain_list,
                _impl::instantiate_local_domain<Dom, ArgList>(m_dom, m_arg_list)
            );
        }

    private:
        Dom const& m_dom;
        ArgList const& m_arg_list;
    };

    template <typename Index>
    struct has_index_ {
        typedef static_int<Index::value> val1;
        template <typename Elem>
        struct apply {
            typedef static_int<Elem::second::value> val2;

            typedef typename boost::is_same<val1, val2>::type type;
        };
    };

    /**@brief metafunction for accessing the storage given the list of placeholders and the temporary pairs list.
       default template parameter: because in case we don't have temporary storages there's no need to specify the pairs.*/
    template <typename Placeholders,
    typename TmpPairs=boost::mpl::na>
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

            template <uint_t I, uint_t J, uint_t K, uint_t L>
            void operator()(range<I,J,K,L> const&) const {
                std::cout << prefix << range<I,J,K,L>() << std::endl;
            }

            template <typename MplVector>
            void operator()(MplVector const&) const {
                // std::cout << "Independent" << std::endl;
                // //gridtools::for_each<MplVector>(print__(std::string("    ")));
                // std::cout << "End Independent" << std::endl;
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

        static uint_t apply(ArgListType& storage_pointers, DomainType &  domain){
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
        static int_t apply(ArgListType const& storage_pointers, DomainType &  domain){
            return GT_NO_ERRORS;
        }
    };

    // /** runtime function applying a lambda/functor with 2 arguments a const number of times (e.g. assignment of constant-sized arrays) */
    //       template <int iterator >
    //           struct associate
    //       {
    //           template<typename FieldType, template<int idx=iterator> class F(FieldType const& , FieldType& )>
    //           void assign(FieldType const& from, FieldType & to){
    //               F<iterator>(from, to);
    //               associate::assign<iterator-1>(from, to);
    //           }
    //       };

    //       template<> struct associate{template <typename FieldType> void apply(FieldType const& from, FieldType & to){F<0>(from,to);}}

/**
 * @class
*  @brief structure collecting helper metafunctions
 * */
    template <typename Backend, typename LayoutType, typename TMssArray, typename DomainType, typename Coords>
    struct intermediate : public computation {

        BOOST_STATIC_ASSERT((is_meta_array<TMssArray>::value));

        /**
         * Takes the domain list of storage pointer types and transform
         * the no_storage_type_yet with the types provided by the
         * backend with the interface that takes the range sizes. This
         * must be done before getting the local_domain
         */
        typedef typename Backend::template obtain_temporary_storage_types<
            DomainType,
            TMssArray,
            float_type,
            /*layout_map<0,1,2>*/LayoutType
        >::type mpl_actual_tmp_pairs;

        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename DomainType::placeholders>::type::value> iter_range;

        typedef typename boost::mpl::fold<
            iter_range,
            boost::mpl::vector0<>,
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


        typedef typename boost::mpl::transform<
            typename TMssArray::elements_t,
            mss_local_domain<boost::mpl::_, DomainType, actual_arg_list_type>
        >::type MssLocalDomains;

        typedef typename boost::fusion::result_of::as_vector<MssLocalDomains>::type MssLocalDomainsList;

        MssLocalDomainsList mss_local_domain_list;

//        /**
//         * Create a fusion::vector of domains for each functor
//         *
//         */
//        typedef typename boost::mpl::transform<
//            typename MssType::linear_esf,
//            _impl::get_local_domain<actual_arg_list_type, typename DomainType::iterator_list, local_domain> >::type mpl_local_domain_list;
//
//        /**
//         *
//         */
//        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;
//
//        /**
//         *
//         */
//        LocalDomainList local_domain_list;


        DomainType & m_domain;
        const Coords& m_coords;

        actual_arg_list_type actual_arg_list;

        struct printtypes {
            template <typename T>
            void operator()(T *) const {
                std::cout << T() << "            ----------" << std::endl;
            }
        };

        intermediate(DomainType & domain, Coords const & coords)
            : m_domain(domain)
            , m_coords(coords)
        {
            // Each map key is a pair of indices in the axis, value is the corresponding method interval.

#ifndef NDEBUG
#ifndef __CUDACC__
//TODO redo
//            std::cout << "Actual loop bounds ";
//            gridtools::for_each<loop_intervals_t>(_debug::show_pair<Coords>(coords));
//            std::cout << std::endl;
#endif
#endif

            // Extract the ranges from functors to determine iteration spaces bounds

            // For each functor collect the minimum enclosing box of the ranges for the arguments

#ifndef NDEBUG
//TODO redo
//            std::cout << "ranges list" << std::endl;
//            gridtools::for_each<ranges_list>(_debug::print__());
#endif

#ifndef NDEBUG
//TODO redo
//            std::cout << "range sizes" << std::endl;
//            gridtools::for_each<structured_range_sizes>(_debug::print__());
//            std::cout << "end1" <<std::endl;
#endif

#ifndef NDEBUG
//TODO redo
//            gridtools::for_each<range_sizes>(_debug::print__());
//            std::cout << "end2" <<std::endl;
#endif

        //filter the non temporary storages among the storage pointers in the domain
            typedef boost::fusion::filter_view<typename DomainType::arg_list,
                is_storage<boost::mpl::_1> > t_domain_view;

        //filter the non temporary storages among the placeholders passed to the intermediate
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
        virtual void ready () {
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
        virtual void steady () {
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

            boost::fusion::for_each(mss_local_domain_list,
                _impl::instantiate_mss_local_domain<DomainType, actual_arg_list_type>(m_domain, actual_arg_list));

#ifndef NDEBUG
            m_domain.info();
#endif
        }

        virtual void finalize () {
            finalize_computation<Backend::s_backend_id>::apply(m_domain);

            // The code below segfaults with CUDA
            // //DELETE the TEMPORARIES (a shared_ptr would be way better)
            // typedef boost::fusion::filter_view<actual_arg_list_type,
            //     is_temporary_storage<boost::mpl::_1> > view_type;
            // view_type fview(actual_arg_list);
            // boost::fusion::for_each(fview, _impl::delete_tmps());
        }

        template<
            typename t_coords,
            typename t_mss_local_domains_list,
            typename mss_array>
        struct run_backend_functor
        {
            run_backend_functor(const t_coords& coords, t_mss_local_domains_list& mss_local_domains_list) :
                m_coords(coords), m_mss_local_domains_list(mss_local_domains_list){}

            template<typename TIndex>
            void operator()(TIndex&) const {

                typedef typename boost::mpl::at<mss_array, TIndex>::type MssType;
                std::cout << "REUNING BACKEND " << typeid(MssType).name() << std::endl;

                Backend::template run<MssType>( m_coords, boost::fusion::at<TIndex>(m_mss_local_domains_list).local_domain_list );
                std::cout << "OUT" << std::endl;
            }

        private:
            const t_coords& m_coords;
            t_mss_local_domains_list& m_mss_local_domains_list;
        };

        /**
         * \brief the execution of the stencil operations take place in this call
         *
         */
        virtual void run () {
            // gridtools::for_each<typename DomainType::placeholders>(_print_____());
            // gridtools::for_each<temp_list>(_print______());
            // std::cout << "---" << std::endl;
            // gridtools::for_each<tomp_list>(_print_____());
            // std::cout << "--- ---" << std::endl;
            // gridtools::for_each<mpl_local_domain_list>(_print_____());

#ifndef NDEBUG
            boost::fusion::for_each(actual_arg_list, _debug::_print_the_storages());
#endif

            BOOST_STATIC_ASSERT((boost::mpl::size<typename TMssArray::elements_t>::value == boost::mpl::size<MssLocalDomains>::value));

            Backend::template run<TMssArray>( m_coords, mss_local_domain_list );

//            gridtools::for_each<boost::mpl::range_c<int, 0, boost::mpl::size<TMssArray>::value > >
//                (run_backend_functor<Coords, MssLocalDomainsList, TMssArray>(m_coords, mss_local_domain_list));
        }

    private:
        bool is_storage_ready;
    };

} // namespace gridtools
