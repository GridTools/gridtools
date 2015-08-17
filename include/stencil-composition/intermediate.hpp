#pragma once
#include "make_stencils.hpp"
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
#include "level.hpp"
#include "loopintervals.hpp"
#include "functor_do_methods.hpp"
#include "functor_do_method_lookup_maps.hpp"
#include "axis.hpp"
#include "local_domain.hpp"
#include "computation.hpp"
#include "heap_allocated_temps.hpp"
#include "mss_local_domain.hpp"
#include "common/meta_array.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "mss_components_metafunctions.hpp"
#include "../storage/storage_functors.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */

namespace gridtools {
    namespace _impl{

        /** @brief Functor used to instantiate the local domains to be passed to each
            elementary stencil function */
        template <typename ArgList, bool IsStateful>
        struct instantiate_local_domain {
            GT_FUNCTION
            instantiate_local_domain(ArgList const& arg_list)
                : m_arg_list(arg_list)
            {}

            /**Elem is a local_domain*/
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<Elem>::value), "Internal Error: wrong type");

                elem.init(m_arg_list, 0,0,0);
                elem.clone_to_gpu();
            }

        template <typename StoragePointers,
                  template <class A, class C, bool B> class LocalDomain>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef LocalDomain<StoragePointers,T,IsStateful> type;
            };
        };

        private:
            ArgList const& m_arg_list;
        };


        /** @brief Functor used to instantiate the local domains to be passed to each
            elementary stencil function */
        template <typename ArgList, bool IsStateful>
        struct instantiate_mss_local_domain {
            GT_FUNCTION
            instantiate_mss_local_domain(ArgList const& arg_list)
                : m_arg_list(arg_list)
            {}

            /**Elem is a local_domain*/
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & mss_local_domain_list) const {
                GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain<Elem>::value), "Internal Error: wrong type");

                boost::fusion::for_each(mss_local_domain_list.local_domain_list,
                                        _impl::instantiate_local_domain<ArgList, IsStateful>(m_arg_list)
                                        );
            }

        private:
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

            template <int_t I, int_t J, int_t K, int_t L, int_t M, int_t N>
            void operator()(range<I,J,K,L,M,N> const&) const {
                std::cout << prefix << range<I,J,K,L,M,N>() << std::endl;
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

    } // namespace _debug

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

            //copy pointers into the domain original pointers, except for the temporaries.
            gridtools::for_each<
                boost::mpl::range_c<int, 0, boost::mpl::size<ArgListType>::value >
            > (copy_pointers_functor<ArgListType, typename DomainType::arg_list> (storage_pointers, domain.original_pointers));

            boost::fusion::for_each(storage_pointers, update_pointer());
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

    /**
     * @brief metafunction that create the mss local domain type
     */
    template<
        enumtype::backend BackendId,
        typename MssComponentsArray,
        typename DomainType,
        typename ActualArgListType,
        bool IsStateful
    > struct create_mss_local_domains
    {
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");

        struct get_the_mss_local_domain {
            template <typename T>
            struct apply {
                typedef mss_local_domain<BackendId, T, DomainType, ActualArgListType, IsStateful> type;
            };
        };

        typedef typename boost::mpl::transform<
            typename MssComponentsArray::elements,
            get_the_mss_local_domain
        >::type type;
    };

    /**
     * @brief computes the list of actual arg types by replacing the temporaries with their
     * actual storage type
     */
    template<
        typename Backend,
        typename DomainType,
        typename MssComponentsArray,
        typename StencilValueType,
        typename LayoutType
    >
    struct create_actual_arg_list
    {
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");

        /**
         * Takes the domain list of storage pointer types and transform
         * the no_storage_type_yet with the types provided by the
         * backend with the interface that takes the range sizes. This
         * must be done before getting the local_domain
         */
        typedef typename Backend::template obtain_temporary_storage_types<
            DomainType,
            MssComponentsArray,
            StencilValueType,
            LayoutType
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

        typedef typename boost::fusion::result_of::as_vector<mpl_actual_arg_list>::type type;

    };

    /**
     * @class
     *  @brief structure collecting helper metafunctions
     */
    template <typename Backend,
              typename LayoutType,
              typename MssDescriptorArray,
              typename DomainType,
              typename Coords,
              bool IsStateful>
    struct intermediate : public computation {
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssDescriptorArray, is_mss_descriptor>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_backend<Backend>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_layout_map<LayoutType>::value), "Internal Error: wrong type");

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;

        typedef typename boost::mpl::fold<
            typename MssDescriptorArray::elements,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                mss_compute_range_sizes<boost::mpl::_2>
            >
        >::type range_sizes_t;

        // build the meta array with all the mss components
        typedef typename build_mss_components_array<
            backend_id<Backend>::value,
            MssDescriptorArray,
            range_sizes_t
        >::type mss_components_array_t;

        typedef typename create_actual_arg_list<
                Backend,
                DomainType,
                mss_components_array_t,
                float_type,
                LayoutType
        >::type actual_arg_list_type;

        typedef typename create_mss_local_domains<
            backend_id<Backend>::value,
            mss_components_array_t,
            DomainType,
            actual_arg_list_type,
            IsStateful
        >::type mss_local_domains_t;

        typedef typename boost::fusion::result_of::as_vector<mss_local_domains_t>::type MssLocalDomainsList;

        MssLocalDomainsList mss_local_domain_list;

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
            : m_domain(domain), m_coords(coords), m_meter("NoName")
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

            boost::fusion::copy(domain_view, args_view);
        }
        /**
           @brief This method allocates on the heap the temporary variables.
           Calls heap_allocated_temps::prepare_temporaries(...).
           It allocates the memory for the list of ranges defined in the temporary placeholders.
        */
        virtual void ready () {
            Backend::template prepare_temporaries( actual_arg_list, m_coords);
            is_storage_ready=true;
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
#ifdef __VERBOSE__
                    printf("Setup computation\n");
#endif
            }
            else
            {
                    printf("Setup computation FAILED\n");
                    exit( GT_ERROR_NO_TEMPS );
            }

            boost::fusion::for_each(mss_local_domain_list,
                   _impl::instantiate_mss_local_domain<actual_arg_list_type, IsStateful>(actual_arg_list));

#ifdef __VERBOSE__
            m_domain.info();
#endif
        }

        virtual void finalize () {
            finalize_computation<Backend::s_backend_id>::apply(m_domain);

            // The code below segfaults with CUDA
//#ifndef __CUDACC__
            //DELETE the TEMPORARIES (a shared_ptr would be way better)
            typedef boost::fusion::filter_view<actual_arg_list_type,
                is_temporary_storage<boost::mpl::_1> > view_type;
            view_type fview(actual_arg_list);
            boost::fusion::for_each(fview, _impl::delete_tmps());
//#endif
        }

        /**
         * \brief the execution of the stencil operations take place in this call
         *
         */
        virtual void run () {

            GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size<typename mss_components_array_t::elements>::value == boost::mpl::size<mss_local_domains_t>::value),
                    "Internal Error");

            m_meter.start();
            Backend::template run<mss_components_array_t>( m_coords, mss_local_domain_list );
            m_meter.pause();
        }

        virtual std::string print_meter() { return m_meter.to_string();}

    private:
        bool is_storage_ready;
        performance_meter_t m_meter;
    };

} // namespace gridtools
