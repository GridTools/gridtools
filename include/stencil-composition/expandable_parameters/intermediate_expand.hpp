#pragma once
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>

#include "../intermediate.hpp"
#include "../../storage/storage.hpp"

#include "intermediate_expand_metafunctions.hpp"
namespace gridtools{

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
                            _impl::is_expandable_arg<boost::mpl::_2>
                            , _impl::create_arg<boost::mpl::_2, ExpandFactor >
                            , boost::mpl::_2
                            >
                      >
                >::type new_arg_list;

            typedef typename boost::mpl::fold<
                new_arg_list
                , boost::mpl::vector0<>
                , boost::mpl::push_back<boost::mpl::_1, pointer<_impl::get_storage<boost::mpl::_2> > >
                >::type new_storage_list;

            typedef typename boost::mpl::fold<
                typename DomainType::placeholders_t
                , boost::mpl::vector0<>
                , boost::mpl::if_<
                      _impl::is_expandable_arg<boost::mpl::_2>
                      , boost::mpl::push_back<
                            boost::mpl::_1
                            , boost::mpl::_2>
                      >
                >::type expandable_p_vector;


            typedef typename boost::mpl::fold<
                typename DomainType::placeholders_t
                , boost::mpl::vector0<>
                , boost::mpl::if_<
                      _impl::is_expandable_arg<boost::mpl::_2>
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


        typedef typename boost::fusion::result_of::as_vector<new_storage_list>::type vec_t;

        // public methods
        intermediate_expand(DomainType &domain, Grid const &grid, ConditionalsSet conditionals_):
            m_domain_from(domain)
            , m_domain_to()
            , m_intermediate()
            , m_intermediate_extra()
        {

            vec_t vec;
            boost::mpl::for_each<expandable_params_t>(_impl::initialize_storage<DomainType, vec_t >( domain, vec));

            // doing the right thing because of the overload of operator= in
            // the expandable_parameter storage
            boost::fusion::copy(domain.m_storage_pointers, vec);

            m_domain_to.reset(new domain_type<new_arg_list>(vec));
            m_intermediate.reset(new intermediate_t(*m_domain_to, grid, conditionals_));
            if(size_%ExpandFactor::value)
                m_intermediate_extra.reset(new intermediate_extra_t(*m_domain_to, grid, conditionals_));

            boost::mpl::for_each<expandable_params_t>(_impl::delete_storage<vec_t >(vec));
        }

        virtual void run(){

            for(uint_t i=0; i<size_-size_%ExpandFactor::value; i+=ExpandFactor::value){

                std::cout<<"iteration: "<<i<<"\n";
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, i));
                // new_domain_.get<ExpandFactor::arg_t>()->assign_pointers(domain.get<ExpandFactor::arg_t>(), i);
                m_intermediate->run();
            }

            if(size_%ExpandFactor::value)
            {
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, size_-size_%ExpandFactor::value));

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
