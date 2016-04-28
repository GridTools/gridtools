#pragma once

/** @file metafunctions used in @ref gridtools::intermediate_expand*/

namespace gridtools{
    namespace _impl{

// ********* metafunctions ************
        template<typename T>
        struct is_expandable_parameters : boost::mpl::false_{};

        template<typename BaseStorage, ushort_t N>
        struct is_expandable_parameters<expandable_parameters<BaseStorage, N> > : boost::mpl::true_{};

        template<typename BaseStorage>
        struct is_expandable_parameters<std::vector<pointer<BaseStorage> > > : boost::mpl::true_{};

        template<typename T>
        struct is_expandable_arg : boost::mpl::false_{};

        template<uint_t N, typename Storage, typename Condition>
        struct is_expandable_arg <arg<N, Storage, Condition > > : is_expandable_parameters<Storage>{};

        template<uint_t N, typename Storage>
        struct is_expandable_arg <arg<N, Storage> > : is_expandable_parameters<Storage>{};

        template <typename T>
        struct get_basic_storage{
            typedef typename T::storage_type::basic_type type;
        };

        template <ushort_t ID, typename T>
        struct get_basic_storage< arg<ID, std::vector<pointer<T> > > >{
            typedef typename T::basic_type type;
        };

        template <typename T>
        struct get_storage{
            typedef typename T::storage_type type;
        };

        template <typename T>
        struct get_index{
            typedef typename T::index_type type;
            static const uint_t value = T::index_type::value;
        };

        template< enumtype::platform B>
        struct create_arg;

        template< >
        struct create_arg<enumtype::Host>
        {
            template<typename T, typename ExpandFactor>
            struct apply{
            typedef arg<get_index<T>::value, expandable_parameters<
                                                 typename get_basic_storage<T>::type
                                                 , ExpandFactor::value>
                        > type;
            };
        };

        template< >
        struct create_arg<enumtype::Cuda>
        {
            template<typename T, typename ExpandFactor>
            struct apply{
            typedef arg<get_index<T>::value, storage<expandable_parameters<
                                                 typename get_basic_storage<T>::type
                                                         , ExpandFactor::value> >
                        > type;
            };
        };


/**
   @brief functor used to initialize the storage in a boost::fusion::vector from an
   instance of gridtools::domain_type
*/
        template<typename DomainFrom, typename Vec>
        struct initialize_storage{

        private:

            DomainFrom const& m_dom_from;
            Vec& m_vec_to;

        public:
            initialize_storage(DomainFrom const& dom_from_, Vec & vec_to_)
                :
                m_dom_from(dom_from_)
                , m_vec_to(vec_to_){}

            template <typename T>
            void operator()(T){
                boost::fusion::at<typename T::index_type>(m_vec_to)
                    =
                    new
                    typename boost::remove_reference<
                        typename boost::fusion::result_of::at<
                            Vec
                            , typename T::index_type
                            >::type>::type::value_type (m_dom_from.template storage_pointer<T>()->meta_data(), "expandable params", false /*do_allocate*/);
            }

            /**
               @brief initialize the storage vector
             */
            template <ushort_t ID, typename T>
            void operator()(arg<ID,std::vector<pointer<T> > >){
                boost::fusion::at<static_ushort<ID> >(m_vec_to)
                    =
                    new
                    typename boost::remove_reference<
                        typename boost::fusion::result_of::at<
                            Vec
                            , static_ushort<ID>
                            >::type>::type::value_type (
                                m_dom_from.
                                template storage_pointer<
                                arg<ID,std::vector<pointer<T> > > >()->at(0)->meta_data()
                                , "expandable params"
                                , false /*do_allocate*/);
            }

        };

        template<typename Domain>
        struct check_length{

        private :
            Domain & m_domain;
            uint_t m_size;

        public:
            check_length(Domain & dom_, uint_t size_):
                m_domain(dom_)
                , m_size(size_)
            {}

            template <typename Arg>
            void operator() (Arg) const {
                // error here means that the sizes of the expandable parameter lists do not match
                assert(boost::fusion::at<typename Arg::index_type>
                       (m_domain.m_storage_pointers)->size()==m_size);
            }
        };

/**
   @brief functor used to delete the storages containing the chunk of pointers
*/
        template<typename Vec>
        struct delete_storage{

        private:

            Vec& m_vec_to;

        public:
            delete_storage(Vec & vec_to_)
                :
                m_vec_to(vec_to_){}

            template <typename T>
            void operator()(T){
                // setting the flag "externally_managed" in order to avoid that the storage pointers
                // get deleted twice (once here and again when destructing the user-defined storage)
                boost::fusion::at<typename T::index_type>(m_vec_to)->set_externally_managed(true);
                delete_pointer deleter;
                deleter(boost::fusion::at<typename T::index_type>(m_vec_to));
            }
        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template<typename DomainFrom, typename DomainTo>
        struct assign_expandable_params{

        private:

            DomainFrom const& m_dom_from;
            DomainTo& m_dom_to;
            uint_t const& m_idx;

        public:

            assign_expandable_params(DomainFrom const & dom_from_, DomainTo & dom_to_, uint_t const& i_):m_dom_from(dom_from_), m_dom_to(dom_to_),  m_idx(i_){}

            template < ushort_t ID, typename T >
            void operator()(arg<ID, std::vector<pointer<T> > >){

                //the vector of pointers
                pointer<std::vector<pointer<T> > > const& storage_ptr_
                    = m_dom_from.template storage_pointer< arg<ID, std::vector<pointer<T> > > >();

                boost::fusion::at<static_ushort<ID> >(m_dom_to.m_storage_pointers)->set(*storage_ptr_, m_idx);
                //update the device pointers (not copying the heavy data)
                boost::fusion::at<static_ushort<ID> >(m_dom_to.m_storage_pointers)->clone_to_device();
                //copy the heavy data (should be done by the steady)
                // boost::fusion::at<static_ushort<ID> >(m_dom_to.m_storage_pointers)->h2d_update();
            }
        };

    } //namespace _impl
} //namespace gridtools
