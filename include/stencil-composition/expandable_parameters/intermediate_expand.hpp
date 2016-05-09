#pragma once
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>

#include "../intermediate.hpp"
#include "../../storage/storage.hpp"

#include "intermediate_expand_metafunctions.hpp"
namespace gridtools{


/**
 * @file
 * \brief this file contains the intermediate representation used in case of expandable parameters
 * */


    /**
       @brief the intermediate representation object

       The expandable parameters are long lists of storages on which the same stencils are applied,
       in a Single-Stencil-Multiple-Storage way. In order to avoid resource contemption usually
       it is convenient to split the execution in multiple stencil, each stencil operating on a chunk
       of the list. Say that we have an expandable parameters list of length 23, and a chunk size of
       4, we'll execute 5 stencil with a "vector width" of 4, and one stencil with a "vector width"
       of 3 (23%4).

       This object contains two unique pointers of @ref gridtools::intermediate type, one with a
       vector width
       corresponding to the expand factor defined by the user (4 in the previous example), and another
       one with a vector width of expand_factor%total_parameters (3 in the previous example).
       In case the total number of parameters is a multiple of the expand factor, the second
       intermediate object does not get instantiated.
     */
    template < typename Backend,
               typename MssDescriptorArray,
               typename DomainType,
               typename Grid,
               typename ConditionalsSet,
               typename ReductionType,
               bool IsStateful,
               typename ExpandFactor
               >
    struct intermediate_expand : public computation<ReductionType>
    {
        GRIDTOOLS_STATIC_ASSERT((is_backend<Backend>::value), "wrong type");
        // to make the following work we should change in lot of places
        // is_mss_descriptor with is_computation_token
        // GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssDescriptorArray, is_computation_token >::value)
        //                         , "wrong type" );
        GRIDTOOLS_STATIC_ASSERT((is_domain_type<DomainType>::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_expand_factor<ExpandFactor>::value), "wrong type");

        // create an mpl vector of @ref gridtools::arg, substituting the large
        // expandable parameters list with a chunk
        typedef typename boost::mpl::fold<
            typename DomainType::placeholders_t
            , boost::mpl::vector0<>
            , boost::mpl::push_back<
                  boost::mpl::_1
                  , boost::mpl::if_<
                            _impl::is_expandable_arg<boost::mpl::_2>
                        , typename _impl::create_arg<Backend::s_backend_id>::template apply<boost::mpl::_2, ExpandFactor>
                        , boost::mpl::_2
                        >
                  >
            >::type new_arg_list;

        // generates an mpl::vector containing the storage types from the previous new_arg_list
        typedef typename boost::mpl::fold<
            new_arg_list
            , boost::mpl::vector0<>
            , boost::mpl::push_back<boost::mpl::_1, pointer<_impl::get_storage<boost::mpl::_2> > >
            >::type new_storage_list;

        // generates an mpl::vector of the original (large) expandable parameters storage types
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

        // typedef to the intermediate type associated with the vector length of ExpandFactor::value
        typedef intermediate <Backend
                              , MssDescriptorArray
                              , domain_type<new_arg_list>
                              , Grid
                              , ConditionalsSet
                              , ReductionType
                              , IsStateful
                              , ExpandFactor::value
                              > intermediate_t;


        // typedef to the intermediate type associated with the vector length of s_size%ExpandFactor::value
        typedef intermediate <Backend
                              , MssDescriptorArray
                              , domain_type<new_arg_list>
                              , Grid
                              , ConditionalsSet
                              , ReductionType
                              , IsStateful
                              , 1
                              > intermediate_extra_t;

    private:
        // private members
        DomainType const& m_domain_from;
        std::unique_ptr<domain_type<new_arg_list> > m_domain_to;
        std::unique_ptr<intermediate_t> m_intermediate;
        std::unique_ptr<intermediate_extra_t> m_intermediate_extra;
        ushort_t m_size;

    public:


        typedef typename boost::fusion::result_of::as_vector<new_storage_list>::type vec_t;

        // public methods

        /**
           @brief constructor

           Given expandable parameters with size N, creates other @ref gristools::expandable_parameters storages with dimension given by  @ref gridtools::expand_factor
         */
        intermediate_expand(DomainType &domain, Grid const &grid, ConditionalsSet conditionals_):
            m_domain_from(domain)
            , m_domain_to()
            , m_intermediate()
            , m_intermediate_extra()
            , m_size(0)
        {

            //fusion vector of storage lists
            vec_t vec;

            //initialize the storage list objects, whithout allocating the storage for the data snapshots
            boost::mpl::for_each<expandable_params_t>(_impl::initialize_storage<DomainType, vec_t >( domain, vec));

            auto const& storage_ptr_ = boost::fusion::at<
                typename boost::mpl::at_c<
                    expandable_params_t,0>::type::index_type >(domain.m_storage_pointers);

            m_size=storage_ptr_->size();

            //check (in DEBUG mode) that all the expandable parameter lists have the same size
            boost::mpl::for_each<expandable_params_t>(_impl::check_length<DomainType>(domain, m_size));

            m_domain_to.reset(new domain_type<new_arg_list>(vec));
            m_intermediate.reset(new intermediate_t(*m_domain_to, grid, conditionals_));
            if(m_size%ExpandFactor::value)
                m_intermediate_extra.reset(new intermediate_extra_t(*m_domain_to, grid, conditionals_));

            // // delete the storage structs (contain empty pointers, so not a big deal)
            // boost::mpl::for_each<expandable_params_t>(_impl::delete_storage<vec_t >(vec));
        }

        /**
           @brief run the execution

           This method performs a run for the computation on each chunck of expandable parameters.
           Between two iterations it updates the @ref gridtools::domain_type, so that the storage
           pointers for the current chunck get substituted by the next chunk. At the end of the
           iterations, if the number of parameters is not multiple of the expand factor, the remaining
           chunck of storage pointers is consumed.
         */
        virtual auto run()
            ->decltype(m_intermediate_extra->run())
        {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same<decltype(m_intermediate_extra->run()), notype>::value), "Reduction is not alloewd with expandable parameters");
            //the expand factor must be smaller than the total size of the expandable parameters list
            assert(m_size >= ExpandFactor::value);

        for(uint_t i=0; i<m_size-m_size%ExpandFactor::value; i+=ExpandFactor::value){

                // std::cout<<"iteration: "<<i<<"\n";
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, i));
                // new_domain_.get<ExpandFactor::arg_t>()->assign_pointers(domain.get<ExpandFactor::arg_t>(), i);
                m_intermediate->run();
            }
            for(uint_t i=0; i<m_size%ExpandFactor::value; ++i)
            {
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, m_size-m_size%ExpandFactor::value+i));
                m_intermediate_extra->run();
            }

        }


        /**
           @brief forwards to the m_intermediate member

           does not take into account the extra kernel executed when the number of parameters is
           not multiple of the expand factor
         */
        virtual std::string print_meter(){
            return m_intermediate->print_meter();
        }

        /**
           @brief forward the call to the members
         */
        virtual void ready(){

        for(uint_t i=0; i<m_size-m_size%ExpandFactor::value; i+=ExpandFactor::value){

                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, i));
                // new_domain_.get<ExpandFactor::arg_t>()->assign_pointers(domain.get<ExpandFactor::arg_t>(), i);
                m_intermediate->ready();
            }
            for(uint_t i=0; i<m_size%ExpandFactor::value; ++i)
            {
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, m_size-m_size%ExpandFactor::value+i));
                m_intermediate_extra->ready();
            }
        }

        /**
           @brief forward the call to the members
         */
        virtual void steady(){

        for(uint_t i=0; i<m_size-m_size%ExpandFactor::value; i+=ExpandFactor::value){

                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, i));
                // new_domain_.get<ExpandFactor::arg_t>()->assign_pointers(domain.get<ExpandFactor::arg_t>(), i);
                m_intermediate->steady();
            }
            for(uint_t i=0; i<m_size%ExpandFactor::value; ++i)
            {
                boost::mpl::for_each<expandable_params_t>(_impl::assign_expandable_params<DomainType, domain_type<new_arg_list> >(m_domain_from, *m_domain_to, m_size-m_size%ExpandFactor::value+i));
                m_intermediate_extra->steady();
            }
        }


        /**
           @brief forward the call to the members
         */
        virtual void finalize(){
            // copy pointers back
            boost::mpl::for_each<expandable_params_t>(_impl::finalize_expandable_params< DomainType >(m_domain_from));
            // free the space for temporaries and storage_info
            m_intermediate->finalize();
            // free the space for temporaries and storage_info
            m_intermediate_extra->finalize();
        }
    };
}
