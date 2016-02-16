#pragma once

#include <boost/mpl/transform.hpp>
#include <boost/mpl/for_each.hpp>
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
#include "esf.hpp"
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
#include "stencil-composition/compute_extents_metafunctions.hpp"
#include "stencil-composition/grid.hpp"
#include "grid_traits.hpp"
#include "stencil-composition/wrap_type.hpp"
#include "switch_variable.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */

namespace gridtools {

    template<typename T>
    struct if_condition_extract_index_t;

        namespace _impl{

        /** @brief Functor used to instantiate the local domains to be passed to each
            elementary stencil function */
        template <typename ArgList, typename MetaStorages, bool IsStateful>
        struct instantiate_local_domain {

            //TODO check the type of ArgList
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set<MetaStorages>::value, "wrong type");

            GT_FUNCTION
            instantiate_local_domain(ArgList const& arg_list, MetaStorages const& meta_storages_)
                : m_arg_list(arg_list)
                , m_meta_storages(meta_storages_)
                {}

            /**Elem is a local_domain*/
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<Elem>::value), "Internal Error: wrong type");

                elem.init(m_arg_list, m_meta_storages.sequence_view(), 0,0,0);
                elem.clone_to_device();
            }

        private:
            ArgList const& m_arg_list;
            MetaStorages const& m_meta_storages;
        };


        /** @brief Functor used to instantiate the local domains to be passed to each
            elementary stencil function */
        template <typename ArgList, typename MetaStorages, bool IsStateful>
        struct instantiate_mss_local_domain {

            //TODO add check for ArgList
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set<MetaStorages>::value, "wrong type");

            GT_FUNCTION
            instantiate_mss_local_domain(ArgList const& arg_list, MetaStorages const& meta_storages_)
                : m_arg_list(arg_list)
                , m_meta_storages(meta_storages_)
            {}

            /**Elem is a local_domain*/
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & mss_local_domain_list_) const {
                GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain<Elem>::value), "Internal Error: wrong type");

                boost::fusion::for_each(mss_local_domain_list_.local_domain_list,
                                        _impl::instantiate_local_domain<ArgList, MetaStorages, IsStateful>(m_arg_list, m_meta_storages)
                                        );
            }

        private:
            ArgList const& m_arg_list;
            MetaStorages const& m_meta_storages;
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

            template <bool is_temp, typename Storage, typename tmppairs, typename index>
            struct get_the_type;

            template <typename Storage, typename tmppairs, typename index>
            struct get_the_type<true, Storage, tmppairs,index> {
                typedef typename boost::mpl::find_if<
                    tmppairs,
                    has_index_<index>
                >::type iter;

                GRIDTOOLS_STATIC_ASSERT((!boost::is_same<iter, typename boost::mpl::end<tmppairs>::type >::value),
                    "Could not find a temporary, defined in the user domain_type, in the list of storage types used in all mss/esfs. \n"
                    " Check that all temporaries are actually used in at least one user functor");

                typedef typename boost::mpl::deref<iter>::type::first type;
            };

            template <typename Storage, typename tmppairs, typename index>
            struct get_the_type<false, Storage, tmppairs,index> {
                typedef Storage type;
            };

            template <typename Index>
            struct apply {
                typedef typename boost::mpl::at<Placeholders, Index>::type::storage_type storage_type;
                static const bool b = is_temp<storage_type>::value;
                typedef pointer<typename get_the_type<b, storage_type, TmpPairs, Index>::type> type;
            };
        };

    } //namespace _impl


    namespace _debug {
        template <typename Grid>
        struct show_pair {
            Grid m_grid;

            explicit show_pair(Grid const& grid)
                : m_grid(grid)
            {}

            template <typename T>
            void operator()(T const&) const {
                typedef typename index_to_level<typename T::first>::type from;
                typedef typename index_to_level<typename T::second>::type to;
                std::cout << "{ (" << from() << " "
                          << to() << ") "
                          << "[" << m_grid.template value_at<from>() << ", "
                          << m_grid.template value_at<to>() << "] } ";
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
            void operator()(extent<I,J,K,L,M,N> const&) const {
                std::cout << prefix << extent<I,J,K,L,M,N>() << std::endl;
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
                boost::mpl::for_each<MplVector>(print__(std::string("    ")));
                printf("End Independent*\n");
            }
        };

    } // namespace _debug

    //\todo move inside the traits classes
    template<enumtype::platform>
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
       This functor calls h2d_update on all storages and meta storages, in order to
       get the data prepared in the case of GPU execution.

       Returns 0 (GT_NO_ERRORS) on success
    */
    template<enumtype::platform>
    struct setup_computation;

    template<>
    struct setup_computation<enumtype::Cuda>{

        template<typename ArgListType, typename MetaData, typename DomainType>
        static uint_t apply(ArgListType& storage_pointers, MetaData& meta_data_,  DomainType &  domain){

            //TODO check the type of ArgListType and MetaData
            GRIDTOOLS_STATIC_ASSERT(is_domain_type<DomainType>::value, "wrong domain type");

            //copy pointers into the domain original pointers, except for the temporaries.
            boost::mpl::for_each<
                boost::mpl::range_c<int, 0, boost::mpl::size<ArgListType>::value >
            > (copy_pointers_functor<ArgListType, typename DomainType::arg_list> (storage_pointers, domain.m_original_pointers));

            // boost::fusion::for_each(meta_data_, copy_pointers_set_functor<typename DomainType::metadata_set_t::set_t> (domain.m_metadata_set.sequence_view()));

            boost::fusion::for_each(storage_pointers, update_pointer());
            boost::fusion::for_each(meta_data_, update_pointer());

            return GT_NO_ERRORS;
        }
    };

    template<>
    struct setup_computation<enumtype::Host>{
        template<typename ArgListType, typename MetaData, typename DomainType>
        static int_t apply(ArgListType const& storage_pointers, MetaData const& meta_data_, DomainType &  domain){

            //TODO check the type of ArgListType and MetaData
            GRIDTOOLS_STATIC_ASSERT(is_domain_type<DomainType>::value, "wrong domain type");

            return GT_NO_ERRORS;
        }
    };

    /**
     * @brief metafunction that create the mss local domain type
     */
    template<
        enumtype::platform BackendId,
        typename MssComponentsArray,
        typename DomainType,
        typename ActualArgListType,
        typename ActualMetadataListType,
        bool IsStateful
    > struct create_mss_local_domains
    {

        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");

        GRIDTOOLS_STATIC_ASSERT((is_metadata_set<ActualMetadataListType>::value), "Internal Error: wrong type");

        struct get_the_mss_local_domain {
            template <typename T>
            struct apply {
                typedef mss_local_domain<BackendId, T, DomainType, ActualArgListType, ActualMetadataListType, IsStateful> type;
            };
        };

        typedef typename boost::mpl::transform<
            typename MssComponentsArray::elements,
            get_the_mss_local_domain
        >::type type;
    };

    template<
        enumtype::platform BackendId,
        typename MssArray1,
        typename MssArray2,
        typename Cond,
        typename DomainType,
        typename ActualArgListType,
        typename ActualMetadataListType,
        bool IsStateful
        > struct create_mss_local_domains<BackendId, condition<MssArray1, MssArray2, Cond>, DomainType, ActualArgListType, ActualMetadataListType, IsStateful >
    {
        typedef typename create_mss_local_domains<BackendId, MssArray1, DomainType, ActualArgListType, ActualMetadataListType, IsStateful>::type type1;
        typedef typename create_mss_local_domains<BackendId, MssArray2, DomainType, ActualArgListType, ActualMetadataListType, IsStateful>::type type2;
        typedef condition<type1, type2, Cond> type;
    };


        template<typename T>
        struct storage2metadata;

    /**
     * @brief computes the list of actual arg types by replacing the temporaries with their
     * actual storage type
     */
    template<
        typename Backend,
        typename DomainType,
        typename MssComponentsArray,
        typename StencilValueType
    >
    struct create_actual_arg_list
    {
        // GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");

        /**
         * Takes the domain list of storage pointer types and transform
         * the no_storage_type_yet with the types provided by the
         * backend with the interface that takes the extent sizes. This
         * must be done before getting the local_domain
         */
        typedef typename Backend::template obtain_temporary_storage_types<
            DomainType,
            MssComponentsArray,
            StencilValueType
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

    template <typename IsPresent, typename MssComponentsArray, typename Backend>
    struct run_conditionally;

    /**@brief calls the run method when conditionals are defined

       specialization for when the next MSS is not a conditional
    */
    template <typename MssComponentsArray, typename Backend>
    struct run_conditionally<boost::mpl::true_, MssComponentsArray, Backend>
    {
        template<typename ConditionalSet, typename Grid, typename MssLocalDomainList>
        static void apply(ConditionalSet const& /**/, Grid const& grid_, MssLocalDomainList const& mss_local_domain_list_){
            Backend::template run<MssComponentsArray>( grid_, mss_local_domain_list_ );
        }
    };


    /**
       @brief calls the run method when conditionals are defined

       specialization for when the next MSS is a conditional
     */
    template <typename Array1, typename Array2, typename Cond, typename Backend>
    struct run_conditionally<boost::mpl::true_, condition<Array1, Array2, Cond>, Backend>
    {
        template<typename ConditionalSet, typename Grid, typename MssLocalDomainList>
        static void apply(ConditionalSet const& conditionals_set_, Grid const& grid_, MssLocalDomainList const& mss_local_domain_list_){
            // std::cout<<"true? "<<boost::fusion::at_key< Cond >(conditionals_set_).value()<<std::endl;
            if(boost::fusion::at_key< Cond >(conditionals_set_).value())
            {
                run_conditionally<boost::mpl::true_, Array1, Backend>::apply( conditionals_set_,  grid_, mss_local_domain_list_ );
            }
            else
                run_conditionally<boost::mpl::true_, Array2, Backend>::apply( conditionals_set_, grid_, mss_local_domain_list_ );
        }
    };

    /**@brief calls the run method when no conditional is defined

       the 2 cases are separated into 2 different partial template specialization, because
       the fusion::at_key doesn't compile when the key is not present in the set
       (i.e. the present situation).
     */
    template <typename MssComponentsArray, typename Backend>
    struct run_conditionally<boost::mpl::false_, MssComponentsArray, Backend>
    {
        template<typename ConditionalSet, typename Grid, typename MssLocalDomainList>
        static void apply(ConditionalSet const& , Grid const& grid_, MssLocalDomainList const& mss_local_domain_list_){

            Backend::template run<MssComponentsArray>( grid_, mss_local_domain_list_ );
        }
    };

    template<typename MssDescriptorArray>
    struct compute_extent_sizes{

        typedef typename select_mss_compute_extent_sizes::type mss_compute_extent_sizes_t;

        typedef typename boost::mpl::fold<
            MssDescriptorArray,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                mss_compute_extent_sizes_t::apply<boost::mpl::_2>
            >
        >::type type;

    };

    template<typename Vec>
    struct extract_mss_domains{
        typedef Vec type;
    };

    template<typename Vec1, typename Vec2, typename Cond>
    struct extract_mss_domains<condition<Vec1, Vec2, Cond> >{

        // TODO: how to do the check described below?
        // GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename extract_mss_domains<Vec1>::type, typename extract_mss_domains<Vec2>::type>::type::value), "The case in which 2 different mss are enabled/disabled using conditionals is supported only when they work with the same placeholders. Here you are trying to switch between MSS for which the type (or the order) of the placeholders is not the same");
        //consider the first one
        typedef typename extract_mss_domains<Vec1>::type type;
    };

    template<typename Array1, typename Array2, typename Cond>
    struct compute_extent_sizes<condition<Array1, Array2, Cond> >{

        typedef typename compute_extent_sizes<Array1>::type type1;
        typedef typename compute_extent_sizes<Array2>::type type2;
        typedef condition<type1, type2, Cond> type;
    };

    /**
     * @class
     *  @brief structure collecting helper metafunctions
     */
    template <typename Backend,
              typename MssDescriptorArray,
              typename DomainType,
              typename Grid,
              typename ConditionalsSet,
              bool IsStateful>
    struct intermediate : public computation {

        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssDescriptorArray, is_mss_descriptor>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_backend<Backend>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Internal Error: wrong type");
        // GRIDTOOLS_STATIC_ASSERT((is_conditionals_set<ConditionalsSet>::value), "Internal Error: wrong type");

        typedef ConditionalsSet conditionals_set_t;
        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;

        typedef typename compute_extent_sizes<typename MssDescriptorArray::elements>::type extent_sizes_t;

        typedef typename build_mss_components_array<
            backend_id<Backend>::value,
            MssDescriptorArray,
            extent_sizes_t
        >::type mss_components_array_t;

        typedef typename create_actual_arg_list<
                Backend,
                DomainType,
                mss_components_array_t,
                float_type
        >::type actual_arg_list_type;

        // build the meta storage typelist with all the mss components
        typedef typename boost::mpl::fold<
            actual_arg_list_type
            , boost::mpl::set<>
            , boost::mpl::if_< is_any_storage<boost::mpl::_2>,
                               boost::mpl::insert<boost::mpl::_1, pointer
                                                  <boost::add_const
                                                   <storage2metadata
                                                    <boost::remove_pointer<boost::mpl::_2>
                                                     >
                                                    >
                                                   >
                                                  >,
                               boost::mpl::_1
                               >
                               >::type actual_metadata_set_t;

        /** transform the typelist into an mpl vector */
        typedef typename boost::mpl::fold<
            actual_metadata_set_t
            , boost::mpl::vector0<>
            , boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 > >::type actual_metadata_vector_t;

        /** define the corresponding metadata_set type (i.e. a fusion set)*/
        typedef metadata_set<actual_metadata_set_t> actual_metadata_list_type;

        /** creates an mpl sequence of local domains*/
        typedef typename create_mss_local_domains<
            backend_id<Backend>::value,
            mss_components_array_t,
            DomainType,
            actual_arg_list_type,
            actual_metadata_list_type,
            IsStateful
            >::type mss_local_domains_t;

        /** creates a fusion vector of local domains*/
        typedef typename boost::fusion::result_of::as_vector<typename extract_mss_domains<mss_local_domains_t>::type >::type mss_local_domain_list_t;

        struct printtypes {
            template <typename T>
            void operator()(T *) const {
                std::cout << T() << "            ----------" << std::endl;
            }
        };

    private:
        mss_local_domain_list_t m_mss_local_domain_list;

        DomainType & m_domain;
        const Grid& m_grid;

        actual_arg_list_type m_actual_arg_list;
        actual_metadata_list_type m_actual_metadata_list;

        bool is_storage_ready;
        performance_meter_t m_meter;

        conditionals_set_t m_conditionals_set;

    public:

        intermediate(DomainType & domain, Grid const & grid, ConditionalsSet const& conditionals_)
            : m_domain(domain), m_grid(grid), m_meter("NoName"), m_conditionals_set(conditionals_)
        {
            // Each map key is a pair of indices in the axis, value is the corresponding method interval.

#ifndef NDEBUG
#ifndef __CUDACC__
//TODO redo
//            std::cout << "Actual loop bounds ";
//            gridtools::for_each<loop_intervals_t>(_debug::show_pair<Grid>(grid));
//            std::cout << std::endl;
#endif
#endif

            // Extract the extents from functors to determine iteration spaces bounds

            // For each functor collect the minimum enclosing box of the extents for the arguments

#ifndef NDEBUG
//TODO redo
//            std::cout << "extents list" << std::endl;
//            gridtools::for_each<extents_list>(_debug::print__());
#endif

#ifndef NDEBUG
//TODO redo
//            std::cout << "extent sizes" << std::endl;
//            gridtools::for_each<structured_extent_sizes>(_debug::print__());
//            std::cout << "end1" <<std::endl;
#endif

#ifndef NDEBUG
//TODO redo
//            gridtools::for_each<extent_sizes>(_debug::print__());
//            std::cout << "end2" <<std::endl;
#endif

            //filter the non temporary storages among the storage pointers in the domain
            typedef boost::fusion::filter_view<typename DomainType::arg_list,
                                               is_not_tmp_storage<boost::mpl::_1> > t_domain_view;

            //filter the non temporary storages among the placeholders passed to the intermediate
            typedef boost::fusion::filter_view<actual_arg_list_type,
                                               is_not_tmp_storage<boost::mpl::_1> > t_args_view;

            t_domain_view domain_view(domain.m_storage_pointers);
            t_args_view args_view(m_actual_arg_list);

            boost::fusion::copy(domain_view, args_view);

            //filter the non temporary meta storages among the storage pointers in the domain
            typedef boost::fusion::filter_view<typename DomainType::metadata_ptr_list,
                                               boost::mpl::not_<is_ptr_to_tmp<boost::mpl::_1> > > t_domain_meta_view;

            //filter the non temporary meta storages among the placeholders passed to the intermediate
            typedef boost::fusion::filter_view<typename boost::fusion::result_of::as_set<actual_metadata_set_t>::type,
                                               boost::mpl::not_<is_ptr_to_tmp<boost::mpl::_1> > > t_meta_view;

            t_domain_meta_view  domain_meta_view(domain.m_metadata_set.sequence_view());
            t_meta_view  meta_view(m_actual_metadata_list.sequence_view());

            //get the storage metadatas from the domain_type
            boost::fusion::copy(domain_meta_view, meta_view);

        }
        /**
           @brief This method allocates on the heap the temporary variables.
           Calls heap_allocated_temps::prepare_temporaries(...).
           It allocates the memory for the list of extents defined in the temporary placeholders.
        */
        virtual void ready () {
            Backend::template prepare_temporaries( m_actual_arg_list, m_actual_metadata_list , m_grid);
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
                //filter the non temporary meta storage pointers among the actual ones
                typename boost::fusion::result_of::as_set<actual_metadata_set_t>::type  meta_view(m_actual_metadata_list.sequence_view());

                setup_computation<Backend::s_backend_id>::apply( m_actual_arg_list, meta_view, m_domain );
#ifdef VERBOSE
                printf("Setup computation\n");
#endif
            }
            else
            {
                    printf("Setup computation FAILED\n");
                    exit( GT_ERROR_NO_TEMPS );
            }

            boost::fusion::for_each(m_mss_local_domain_list,
                                    _impl::instantiate_mss_local_domain<actual_arg_list_type, actual_metadata_list_type, IsStateful>(m_actual_arg_list, m_actual_metadata_list));

#ifdef VERBOSE
            m_domain.info();
#endif
        }

        virtual void finalize () {
            finalize_computation<Backend::s_backend_id>::apply(m_domain);

            //DELETE the TEMPORARIES (a shared_ptr would be way better)
            //NOTE: the descrutor of the copy_to_gpu stuff will automatically free the storage
            //on the GPU
            typedef boost::fusion::filter_view<actual_arg_list_type,
                is_temporary_storage<boost::mpl::_1> > view_type;
            view_type fview(m_actual_arg_list);
            boost::fusion::for_each(fview, _impl::delete_tmps());

            //deleting the metadata objects
            typedef boost::fusion::filter_view<typename actual_metadata_list_type::set_t,
                is_ptr_to_tmp<boost::mpl::_1> > view_type2;
            view_type2 fview2(m_actual_metadata_list.sequence_view());
            boost::fusion::for_each(fview2, delete_pointer());
        }

        /**
         * \brief the execution of the stencil operations take place in this call
         *
         */
        virtual void run () {

            // GRIDTOOLS_STATIC_ASSERT(
            //     (boost::mpl::size<typename mss_components_array_t::first>::value == boost::mpl::size<typename mss_local_domains_t::first>::value),
            //     "Internal Error");

            //typedef allowing compile-time dispatch: we separate the path when the first
            //multi stage stencil is a conditional
            typedef typename boost::fusion::result_of::has_key<conditionals_set_t,
                                                               typename if_condition_extract_index_t<
                                                                   mss_components_array_t
                                                                   >::type
                                                               >::type is_present_t;

            m_meter.start();
            run_conditionally<is_present_t, mss_components_array_t, Backend>::apply(m_conditionals_set, m_grid, m_mss_local_domain_list);
            m_meter.pause();
        }

        virtual std::string print_meter() { return m_meter.to_string();}

        mss_local_domain_list_t const& mss_local_domain_list() const {return m_mss_local_domain_list; }
    };


    /**@brief resets the conditional variable used in an if_ statement from whithin a computation*/
    template<uint_t Id>
    void reset_conditional(conditional<Id>& cond_, conditional<Id> const& new_cond_){
        cond_.value()=new_cond_.value();
    }

    /**@brief resets the conditional variables generated by a switch_ statement from whithin a computation*/
    template<uint_t Id, typename T>
    void reset_conditional( switch_variable<Id, T>& cond_, switch_variable<Id, T> const& new_cond_){
        for (int_t i=0; i<cond_.num_conditions(); ++i)
            cond_.conditions()[i]= (new_cond_.value() == cond_.cases()[i]);
    }


} // namespace gridtools
