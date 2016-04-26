#pragma once
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>

#include "../intermediate.hpp"
#include "../../storage/storage.hpp"

namespace gridtools{

    template<typename T>
    struct is_expandable_parameters : boost::mpl::false_{};

    template<typename BaseStorage, ushort_t N>
    struct is_expandable_parameters<expandable_parameters<BaseStorage, N> > : boost::mpl::true_{};

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

    template <typename T>
    struct get_storage{
        typedef typename T::storage_type type;
    };

    template <typename T>
    struct get_value_type{
        typedef typename T::value_type type;
    };

    template <typename T>
    struct get_index{
        typedef typename T::index_type type;
        static const uint_t value = T::index_type::value;
    };

    template< typename T, typename ExpandFactor>
    struct create_arg{
        typedef arg<get_index<T>::value, expandable_parameters<
                                             typename get_basic_storage<T>::type
                                             , ExpandFactor::value>
        > type;
    };

    template < typename Backend,
               typename MssDescriptorArray,
               typename DomainType,
               typename Grid,
               typename ConditionalsSet,
               bool IsStateful,
               typename ExpandFactor
               >
    struct intermediate_expand : public computation
    {



            typedef typename boost::mpl::fold<
                typename DomainType::placeholders_t
                , boost::mpl::vector0<>
                , boost::mpl::push_back<
                      boost::mpl::_1
                      , boost::mpl::if_<
                          is_expandable_arg<boost::mpl::_2>
                            , create_arg<boost::mpl::_2, ExpandFactor >
                            , boost::mpl::_2
                            >
                      >
                >::type new_arg_list;

            typedef typename boost::mpl::fold<
                new_arg_list
                , boost::mpl::vector0<>
                , boost::mpl::push_back<boost::mpl::_1, pointer<get_storage<boost::mpl::_2> > >
                >::type new_storage_list;

            typedef typename boost::mpl::fold<
                typename DomainType::placeholders_t
                , boost::mpl::vector0<>
                , boost::mpl::if_<
                      is_expandable_arg<boost::mpl::_2>
                      , boost::mpl::push_back<
                            boost::mpl::_1
                            , boost::mpl::_2>
                      >
                >::type expandable_p_vector;


            typedef typename boost::mpl::fold<
                typename DomainType::placeholders_t
                , boost::mpl::vector0<>
                , boost::mpl::if_<
                    is_expandable_arg<boost::mpl::_2>
                      , boost::mpl::push_back<
                            boost::mpl::_1
                            , boost::mpl::_2 >
                      , boost::mpl::_1
                      >
                >::type expandable_params_t;

        typedef intermediate <Backend
                              , MssDescriptorArray
                              , domain_type<new_arg_list>
                              , Grid
                              , ConditionalsSet
                              , IsStateful
                              , ExpandFactor::value
                              > intermediate_t;

        static const ushort_t size_=boost::mpl::at_c<expandable_params_t,0>::type::storage_type::field_dimensions;

        typedef intermediate <Backend
                              , MssDescriptorArray
                              , domain_type<new_arg_list>
                              , Grid
                              , ConditionalsSet
                              , IsStateful
                              , size_%ExpandFactor::value
                              > intermediate_extra_t;

    private:
        // private members
        DomainType const& m_domain_from;
        std::unique_ptr<domain_type<new_arg_list> > m_domain_to;
        std::unique_ptr<intermediate_t> m_intermediate;
        std::unique_ptr<intermediate_extra_t> m_intermediate_extra;

    public:


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
                    //TODO: who deletes this new? The domain_type?
                    new
                    typename boost::remove_reference<
                        typename boost::fusion::result_of::at<
                            Vec
                            , typename T::index_type
                            >::type>::type::value_type (m_dom_from.template storage_pointer<T>()->meta_data(), "expandable params", false /*do_allocate*/);
            }
        };



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

        typedef typename boost::fusion::result_of::as_vector<new_storage_list>::type vec_t;

        // public methods
        intermediate_expand(DomainType &domain, Grid const &grid, ConditionalsSet conditionals_):
            m_domain_from(domain)
            , m_domain_to()
            , m_intermediate()
            , m_intermediate_extra()
        {

            vec_t vec;
            boost::mpl::for_each<expandable_params_t>(initialize_storage<DomainType, vec_t >( domain, vec));

            // doing the right thing because of the overload of operator= in
            // the expandable_parameter storage
            boost::fusion::copy(domain.m_storage_pointers, vec);

            m_domain_to.reset(new domain_type<new_arg_list>(vec));
            m_intermediate.reset(new intermediate_t(*m_domain_to, grid, conditionals_));
            if(size_%ExpandFactor::value)
                m_intermediate_extra.reset(new intermediate_extra_t(*m_domain_to, grid, conditionals_));

            boost::mpl::for_each<expandable_params_t>(delete_storage<vec_t >(vec));
        }

        template<typename DomainFrom, typename DomainTo>
        struct assign_expandable_params{

        private:

            DomainFrom const& m_dom_from;
            DomainTo& m_dom_to;
            uint_t const& m_idx;

        public:

            assign_expandable_params(DomainFrom const & dom_from_, DomainTo & dom_to_, uint_t const& i_):m_dom_from(dom_from_), m_dom_to(dom_to_),  m_idx(i_){}

            template <typename T>
            void operator()(T){

                m_dom_to.template storage_pointer<T>()->assign_pointers(*m_dom_from.template storage_pointer<T>(), m_idx);
            }
        };

        virtual void run(){

            for(uint_t i=0; i<size_-size_%ExpandFactor::value; i+=ExpandFactor::value){

                std::cout<<"iteration: "<<i<<"\n";
                boost::mpl::for_each<expandable_params_t>(assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, i));
                // new_domain_.get<ExpandFactor::arg_t>()->assign_pointers(domain.get<ExpandFactor::arg_t>(), i);
                m_intermediate->run();
            }

            if(size_%ExpandFactor::value)
            {
                boost::mpl::for_each<expandable_params_t>(assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, size_-size_%ExpandFactor::value));

                m_intermediate_extra->run();
            }

        }


        virtual std::string print_meter(){
            return m_intermediate->print_meter();
        }

        virtual void ready(){
            m_intermediate->ready();
            if(size_%ExpandFactor::value)
            {
                m_intermediate_extra->ready();
            }
        }

        virtual void steady(){
            m_intermediate->steady();
            if(size_%ExpandFactor::value)
            {
                m_intermediate_extra->steady();
            }
        }

        virtual void finalize(){
            m_intermediate->finalize();
            if(size_%ExpandFactor::value)
            {
                m_intermediate_extra->finalize();
            }
        }
    };
}
