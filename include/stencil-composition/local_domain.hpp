#pragma once

#include <boost/mpl/range_c.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/set.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/utility.hpp>
#include "../common/host_device.hpp"
#include "../common/gpu_clone.hpp"
#include "../common/is_temporary_storage.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "arg.hpp"
#include "../gt_for_each/for_each.hpp"
#include "../storage/storage_metafunctions.hpp"

#include <boost/fusion/include/as_set.hpp>

namespace gridtools {

    namespace local_domain_aux {

        template <typename IndicesList, typename ArgList, typename LocalList >
        struct assign_storage_pointers {

            ArgList const& m_arg_list;
            LocalList & m_local_list;

            GT_FUNCTION_WARNING
            assign_storage_pointers(ArgList const& arg_list_, LocalList & local_list_)
                : m_arg_list(arg_list_)
                , m_local_list(local_list_)
            {
                // GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ArgList, is_storage>::value), "wrong type");
                // GRIDTOOLS_STATIC_ASSERT((is_sequence_of<LocalList, is_arg>::value), "wrong type");
            }

            template <typename Id>
            GT_FUNCTION_WARNING
            void operator()(Id) const {

                typedef typename boost::remove_reference
                    <typename boost::mpl::at
                     <IndicesList, Id>::type>::type::index_type index_t;

                boost::fusion::at_c<Id::value>(m_local_list) =
#ifdef __CUDACC__ // ugly ifdef. TODO: way to remove it?
                    boost::fusion::at_c<index_t::value>(m_arg_list)->gpu_object_ptr;
#else
                boost::fusion::at_c<index_t::value>(m_arg_list);
#endif
            }
        };


        /**
           @class
           @brief functor to assign a boost::fusion associative containers with another one, both
           accessed using a key

           \tparam LocalMetaData output container type
           \tparam ActualMetaData input container type
           It is used here to assign the storage meta data of the current local domain (i.e. one
           user functor) given the actual meta data of the whole computation
         */
        template <typename LocalMetaData, typename ActualMetaData>
        struct assign_fusion_maps {

            ActualMetaData const& m_actual;

            GT_FUNCTION_WARNING
            assign_fusion_maps(ActualMetaData const& actual_)
                :
                m_actual(actual_)
            {}

            /**
               @brief assignment

               \tparam Key the key to access one specific instance in the two containers
               \param local_ the instance getting assigned
             */
            template <typename Key>
            GT_FUNCTION_WARNING
            void operator()(Key& local_) const {
                local_ =
#ifdef __CUDACC__ // ugly ifdef. TODO: way to remove it?
                    (typename Key::value_type *) boost::fusion::at_key<Key>(m_actual)->gpu_object_ptr;
#else
                    boost::fusion::at_key<Key>(m_actual);
#endif
            }
        };

        /** Just extract the storage types. In case of temporaries, these types
            are the storage types containing the storage classes that contains
            the "repositories" of all the perthread containers.
        */
        template <typename StorageList>
        struct extract_types {
            template <typename ElemType>
            struct apply {
                typedef typename boost::remove_reference<
                    typename boost::fusion::result_of::at<StorageList, typename ElemType::index_type>::type
                                                         >::type type;
            };
        };

        /** Just extract the storage types. In case of temporaries, these types
            are the storage types containing the actual storage types used by the
            individual threads. This requires a difference w.r.t. extract_types
            for how to deal with temporaries.

            Since certain modifications happend this metafunction is actually
            identical, in behavior, with extract_types.
        */
        template <typename StorageList>
        struct extract_actual_types {

            template <typename Storage, typename Enable=void>
            struct check_if_temporary : boost::mpl::false_{};

            template <typename Storage>
            struct check_if_temporary<Storage, typename boost::enable_if_c<is_temporary_storage<Storage>::value>::type> {
                typedef Storage type;
            };

            template <typename Storage>
            struct check_if_temporary<Storage, typename boost::disable_if_c<is_temporary_storage<Storage>::value>::type> {
                typedef Storage type;
            };


            template <typename ElemType>
            struct apply {
                typedef typename check_if_temporary<
                    typename boost::remove_reference<
                        typename boost::fusion::result_of::at<StorageList, typename ElemType::index_type>::type
                        >::type
                    >::type type;
            };
        };
    } // namespace gt_aux

    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide coordinates to the functor
     * and one that does not
     *
     */
    template<typename T>
    struct local_domain_base;

    template<typename S,typename M,typename E, bool I> class local_domain;

    template <typename StoragePointers, typename MetaStoragePointers, typename EsfArgs, bool IsStateful>
    struct local_domain_base<local_domain<StoragePointers, MetaStoragePointers, EsfArgs, IsStateful> >
        : public clonable_to_gpu<local_domain<StoragePointers, MetaStoragePointers, EsfArgs, IsStateful> >
    {
        template <typename I>
        struct extract_index{
            typedef typename I::index_type type;
        };

        struct extract_index_lambda{
            template <typename I>
            struct apply{
                typedef typename extract_index<I>::type type;
            };
        };



        typedef local_domain<StoragePointers, MetaStoragePointers, EsfArgs, IsStateful> derived_t;

        typedef local_domain_base<derived_t> this_type;

        typedef EsfArgs esf_args;

        typedef StoragePointers storage_pointers_t;

        typedef MetaStoragePointers meta_storage_pointers_t;

        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<esf_args>::type::value > the_range;

        //! creates a vector of placeholders associated with a linear range
        typedef typename boost::mpl::fold<the_range,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::at<esf_args,  boost::mpl::_2>
                                              >
                                          >::type domain_indices_t;

        /** extracts the static_int indices from the args */
        typedef typename boost::mpl::transform<domain_indices_t,
                                               extract_index_lambda
                                               >::type domain_indices_range_t;


        /** creates a vector of storage types from the StoragePointers sequence */
        typedef typename boost::mpl::fold<domain_indices_t,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              typename local_domain_aux::
                                              extract_types<
                                                  StoragePointers>::template apply<boost::mpl::_2>
                                              >
                                          >::type mpl_storages;

        /** creates a vector of storage types from the StoragePointers sequence */
        typedef typename boost::mpl::fold<domain_indices_t,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              typename local_domain_aux::extract_actual_types<
                                                  StoragePointers>::template apply<boost::mpl::_2>
                                              >
                                          >::type mpl_actual_storages;


        /** creates a vector of storage types from the StoragePointers sequence */
        typedef typename boost::mpl::fold
        <mpl_storages,
         boost::mpl::vector0<>,
         boost::mpl::if_< is_any_storage<boost::mpl::_2>,
                          boost::mpl::push_back<
                              boost::mpl::_1,
                              storage2metadata<
                                  boost::remove_pointer<
                                      boost::mpl::_2 >
                                  >
                              >
                          , boost::mpl::_1 >
         >::type::type local_metadata_mpl_t;


        // convoluted way to filter out the duplicated meta storage types : transform vector to set to map
        typedef typename boost::mpl::fold<boost::mpl::range_c<uint_t, 0, boost::mpl::size<local_metadata_mpl_t>::value>
                                          , boost::mpl::set0<>
                                          , boost::mpl::insert
                                          <boost::mpl::_1,boost::mpl::at
                                            <local_metadata_mpl_t, boost::mpl::_2>
                                           >
                                          >::type storage_metadata_set_t;

        typedef typename boost::mpl::fold<storage_metadata_set_t
                                          , boost::mpl::vector0<>
                                          , boost::mpl::push_back
                                          <boost::mpl::_1, boost::mpl::_2>
                                          >::type storage_metadata_vector_t;

        typedef typename boost::mpl::fold<boost::mpl::range_c<uint_t, 0, boost::mpl::size<storage_metadata_vector_t>::value>
                                          , boost::mpl::map0<>
                                          , boost::mpl::insert
                                          <boost::mpl::_1, boost::mpl::pair
                                           <boost::mpl::at
                                            <storage_metadata_vector_t, boost::mpl::_2>,
                                            boost::mpl::_2
                                            >
                                           >
                                          >::type storage_metadata_map;


        typedef typename boost::fusion::result_of::as_vector<mpl_storages>::type local_args_type;
        typedef typename boost::fusion::result_of::as_vector<mpl_actual_storages>::type actual_args_type;


        /*construct the boost fusion vector of metadata pointers*/
        typedef typename boost::fusion::result_of::as_vector<
            typename boost::mpl::transform<storage_metadata_vector_t, pointer<
                                                                     boost::add_const< boost::mpl::_1> > >::type
            >::type local_metadata_type;

        local_args_type m_local_args;
        local_metadata_type m_local_metadata;

        template <typename Dom, typename IsActuallyClonable, uint_t DUMMY = 0>
        struct pointer_if_clonable {
            static Dom* get(Dom* d) {
                return d;
            }
        };

        template <typename Dom, uint_t DUMMY>
        struct pointer_if_clonable<Dom, boost::true_type, DUMMY> {
            static Dom* get(Dom* d) {
                return d->gpu_object_ptr;
            }
        };

        GT_FUNCTION_WARNING
        local_domain_base() {}

        /**
           @brief implements the global to local assignment

           it assigns the local (i.e. of the current esf) storages/metadatas, from the corresponent
           global (i.e. of the whole computation) values.
        */
        template <typename ActualArgs, typename ActualMetaData>
        GT_FUNCTION
        void init(ActualArgs const& actual_args_, ActualMetaData const& actual_metadata_)
        {

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<domain_indices_t>::type::value == boost::mpl::size<local_args_type>::type::value), "sizes not matching");

            typedef static_uint<boost::mpl::size<domain_indices_t>::type::value>  size_type;
            boost::mpl::for_each
                <boost::mpl::range_c
                 <uint_t, 0, size_type::value> > (
                     local_domain_aux::assign_storage_pointers<domain_indices_t
                     , ActualArgs, local_args_type>(actual_args_, m_local_args));

            //assigns the metadata for all the components of m_local_metadata (global to local)
            boost::fusion::for_each(m_local_metadata, local_domain_aux::assign_fusion_maps<local_metadata_type, ActualMetaData>(actual_metadata_));
        }

        __device__
        local_domain_base(local_domain_base const& other)
            : m_local_args(other.m_local_args)
            , m_local_metadata(other.m_local_metadata)
            { }

        template <typename T>
        void info(T const&) const {
            T::info();
            std::cout << "[" << boost::mpl::at_c<esf_args, T::index_type::value>::type::index_type::value << "] ";
        }

        struct show_local_args_info {
            template <typename T>
            void operator()(T const & e) const {
                e->info();
            }
        };

        GT_FUNCTION
        void info() const {
            std::cout << "        -----v SHOWING LOCAL ARGS BELOW HERE v-----" << std::endl;
            boost::fusion::for_each(m_local_args, show_local_args_info());
            std::cout << "        -----^ SHOWING LOCAL ARGS ABOVE HERE ^-----" << std::endl;
        }
    };

    template <typename T>
    struct is_metadata_set;

    /**
     * This class extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. This version does not provide coordinates
     * to the function operator
     *
     * @tparam StoragePointers The mpl vector of the storage pointer types
     * @tparam MetaData The mpl vector of the meta data pointer types sequence
     * @tparam EsfArgs The mpl vector of the args (i.e. placeholders for the storages)
                       for the current ESF
     * @tparam IsStateful The flag stating if the local_domain is aware of the position in the iteration domain
     */
    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    struct local_domain : public local_domain_base< local_domain<StoragePointers, MetaData, EsfArgs, IsStateful> > {

        GRIDTOOLS_STATIC_ASSERT((is_metadata_set<MetaData>::value),"Local domain contains wrong type for parameter meta storages");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfArgs, is_arg>::value),"Local domain contains wront type for parameter placeholders");
        typedef local_domain_base<local_domain<StoragePointers,MetaData,EsfArgs,IsStateful> > base_type;
        typedef StoragePointers storage_pointers;
        typedef EsfArgs esf_args;

        GT_FUNCTION
        local_domain() {}

        __device__
        local_domain(local_domain const& other)
            : base_type(other)
        {}

        /**
           @brief forwarding to the init of the base class
         */
        template <typename ArgList, typename MetaDataList>
        GT_FUNCTION
        void init(ArgList const& arg_list, MetaDataList const& meta_data_, uint_t, uint_t, uint_t)
        {
            base_type::init(arg_list, meta_data_);
        }

        /**stub methods*/
        GT_FUNCTION
        uint_t i() const {return 1e9; }
        GT_FUNCTION
        uint_t j() const {return 1e9; }
        GT_FUNCTION
        uint_t k() const {return 1e9; }
    };

    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    std::ostream& operator<<(std::ostream& s, local_domain<StoragePointers,MetaData, EsfArgs, IsStateful> const&) {
        return s << "local_domain<stuff>";
    }

    template<typename T> struct is_local_domain : boost::mpl::false_{};

    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    struct is_local_domain<local_domain<StoragePointers, MetaData, EsfArgs, IsStateful> > : boost::mpl::true_{};

    template<typename T> struct local_domain_is_stateful;

    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    struct local_domain_is_stateful<local_domain<StoragePointers, MetaData, EsfArgs, IsStateful> > : boost::mpl::bool_<IsStateful>{};

    template<typename T>
    struct local_domain_esf_args;

    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    struct local_domain_esf_args<local_domain<StoragePointers, MetaData, EsfArgs, IsStateful> >
    {
        typedef EsfArgs type;
    };

    template<typename T>
    struct local_domain_storage_pointers;

    template <typename StoragePointers, typename MetaData, typename EsfArgs, bool IsStateful>
    struct local_domain_storage_pointers<local_domain<StoragePointers, MetaData, EsfArgs, IsStateful> >
    {
        typedef StoragePointers type;
    };

    template <typename LocalDomain>
    struct meta_storage_pointers
    {
        typedef typename LocalDomain::meta_storage_pointers_t type;
    };

} // namespace gridtools
