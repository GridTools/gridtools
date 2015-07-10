#pragma once
#include "iterate_domain_aux.hpp"
#include <boost/fusion/include/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include "expressions.hpp"
#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include "local_domain.hpp"
#include "../common/gt_assert.hpp"

/**@file
   @brief file handling the access to the storage.
   This file implements some of the innermost data access operations of the library and thus it must be highly optimized.
   The naming convention used below distinguishes from the following two concepts:

   - a parameter: is a non-space dimension (e.g. time) such that derivatives are taken in the equations along this dimension.
   - a dimension: with an abuse of notation will denote any physical scalar field contained in the given storage, e.g. a velocity component, the pressure, or the energy. I.e. it is an extra dimension which can appear in the equations only derived in space or with respect to the parameter mentioned above.
   - a storage: is an instance of the storage class, and can contain one or more fields and dimensions. Every dimension consists of one or several snaphsots of the scalar fields
   (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)
   - a data snapshot: is a pointer to one single snapshot. The snapshots are arranged in the storages on a 1D array, regardless of the dimension and snapshot they refer to. The accessor (or offset_tuple) class is
   responsible of computing the correct offests (relative to the given dimension) and address the storages correctly.

   The access to the storage is performed in the following steps:
   - the addresses of the first element of all the data fields in the storages involved in this stencil are saved in an array (m_storage_pointers)
   - the index of the storages is saved in another array (m_index)
   - when the functor gets called, the 'offsets' become visible (in the perfect worls they could possibly be known at compile time). In particular the index is moved to point to the correct address, and the correct data snapshot is selected.

   Graphical (ASCII) illustration:

   \verbatim
   ################## Storage #################
   #                ___________\              #
   #                  width    /              #
   #              | |*|*|*|*|*|*|    dim_1    #
   #   dimensions | |*|*|*|          dim_2    #
   #              v |*|*|*|*|*|      dim_3    #
   #                                          #
   #                 ^ ^ ^ ^ ^ ^              #
   #                 | | | | | |              #
   #                 snapshots                #
   #                                          #
   ################## Storage #################
   \endverbatim

*/

namespace gridtools {


    template< typename Impl>
    struct iterate_domain_impl_local_domain;

    template< typename LocalDomain,
        template<typename> class IterateDomainBase,
        template<template<typename> class, typename> class IterateDomainImpl >
    struct iterate_domain_impl_local_domain < IterateDomainImpl<IterateDomainBase, LocalDomain> >
    {
        GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");
        typedef LocalDomain type;
    };

    /**@brief class handling the computation of the */
    template <typename IterateDomainImpl>
    struct iterate_domain {
        typedef typename iterate_domain_impl_local_domain<IterateDomainImpl>::type local_domain_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain<local_domain_t>::value), "Internal Error: wrong type");
        typedef typename boost::remove_pointer<
            typename boost::mpl::at_c<
                typename local_domain_t::mpl_storages, 0>::type
            >::type::value_type value_type;

        //typedef typename local_domain_t::local_args_type local_args_type;
        typedef typename local_domain_t::actual_args_type actual_args_type;
        //the number of storages  used in the current functor
        static const uint_t N_STORAGES=boost::mpl::size<actual_args_type>::value;
        //the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS=total_storages<
            actual_args_type,
            boost::mpl::size<typename local_domain_t::mpl_storages>::type::value >::value;

        typedef array<void* RESTRICT, N_DATA_POINTERS> data_pointer_array_t;
        typedef strides_cached<N_STORAGES-1, typename local_domain_t::esf_args> strides_cached_t;

    private:
        // iterate_domain remembers the state. This is necessary when
        // we do finite differences and don't want to recompute all
        // the iterators (but simply use the ones available for the
        // current iteration storage for all the other storages)

        local_domain_t const& local_domain;
        array<int_t,N_STORAGES> m_index;
        data_pointer_array_t* RESTRICT m_data_pointer;

        strides_cached_t* RESTRICT m_strides;

    public:

        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fileds)
        */
        GT_FUNCTION
        iterate_domain(local_domain_t const& local_domain_)
            : local_domain(local_domain_)
            ,m_data_pointer(0)
            ,m_strides(0) {}

        const void* data_pointer(ushort_t i){return (*m_data_pointer)[i];}

        /** This functon set the addresses of the data values  before the computation
            begins.

            The EU stands for ExecutionUnit (thich may be a thread or a group of
            threasd. There are potentially two ids, one over i and one over j, since
            our execution model is parallel on (i,j). Defaulted to 1.
        */
        template<typename BackendType>
        GT_FUNCTION
        void assign_storage_pointers( data_pointer_array_t* RESTRICT data_pointer ){
            assert(data_pointer);
            const uint_t EU_id_i = BackendType::processing_element_i();
            const uint_t EU_id_j = BackendType::processing_element_j();
            m_data_pointer=data_pointer;
            for_each<typename reversed_range< int_t, 0, N_STORAGES >::type > (
                assign_storage_functor<
                    BackendType,
                    data_pointer_array_t,
                    typename local_domain_t::local_args_type
                >(*m_data_pointer, local_domain.local_args,  EU_id_i, EU_id_j));
        }

        template<typename BackendType, typename Strides>
        GT_FUNCTION
        void assign_stride_pointers( Strides * RESTRICT strides_){
            GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), "internal error type");
            assert(strides_);
            m_strides=strides_;
            for_each< typename reversed_range<int_t, 0, N_STORAGES >::type > (
                assign_strides_functor<
                    BackendType,
                    Strides,
                    typename local_domain_t::local_args_type
                >(*m_strides, local_domain.local_args));
        }


        /**@brief getter for the index array */
        GT_FUNCTION
        void get_index(array<int_t, N_STORAGES>& index) const
            {
                set_index_recur< N_STORAGES-1>::set(m_index, index);
            }

        /**@brief method for setting the index array */
        template <typename Input>
        GT_FUNCTION
        void set_index(Input const& index)
        {
            set_index_recur< N_STORAGES-1>::set( index, m_index);
        }

        /**@brief method for incrementing by 1 the index when moving forward along the given direction
           \tparam Coordinate dimension being incremented
           \tparam Execution the policy for the increment (e.g. forward/backward)
         */
        template <ushort_t Coordinate, typename Execution>
        GT_FUNCTION
        void increment()
        {
            for_each<typename reversed_range< int_t, 0, N_STORAGES >::type > (
                increment_index_functor<
                    Coordinate,
                    strides_cached_t,
                    typename local_domain_t::local_args_type
                >(local_domain.local_args,
#ifdef __CUDACC__ //stupid nvcc
                  boost::is_same<Execution, static_int<1> >::type::value? 1 : -1
#else
                  Execution::value
#endif
                  , &m_index[0], *m_strides)
                );
        }

        /**@brief method for incrementing the index when moving forward along the given direction

           \param steps_ the increment
           \tparam Coordinate dimension being incremented
         */
        template <ushort_t Coordinate>
        GT_FUNCTION
        void increment(int_t steps_)
        {
            for_each<typename reversed_range<int_t, 0, N_STORAGES >::type > (
                increment_index_functor<
                    Coordinate,
                    strides_cached_t,
                    typename local_domain_t::local_args_type
                >(local_domain.local_args, steps_, &m_index[0], *m_strides)
            );
        }

        /**@brief method for initializing the index */
        template <ushort_t Coordinate>
        GT_FUNCTION
        void initialize(uint_t const initial_pos=0, uint_t const block=0)
        {
            for_each<typename reversed_range<int_t, 0, N_STORAGES >::type > (
                initialize_index_functor<
                    Coordinate,
                    strides_cached_t,
                    typename local_domain_t::local_args_type
                >(*m_strides, local_domain.local_args, initial_pos, block, &m_index[0])
            );
        }

        template <typename T>
        GT_FUNCTION
        void info(T const &x) const
            {
                local_domain.info(x);
            }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
           \param arg placeholder containing the storage ID and the offsets
           \param storage_pointer pointer to the first element of the specific data field used
        */
        template <typename ArgType, typename StoragePointer>
        GT_FUNCTION
        typename boost::mpl::at<typename local_domain_t::esf_args, typename ArgType::index_type>::type::value_type& RESTRICT
        get_value(ArgType const& arg , StoragePointer & RESTRICT storage_pointer) const;


        /**@brief local class instead of using the inline (cond)?a:b syntax, because in the latter both branches get compiled (generating sometimes a compile-time overflow) */
        template <bool condition, typename LocalD, typename ArgType>
        struct current_storage;

        template < typename LocalD, typename ArgType>
        struct current_storage<true, LocalD, ArgType>{
            static const uint_t value=0;
        };

        template < typename LocalD, typename ArgType>
        struct current_storage<false, LocalD, ArgType>{
            static const uint_t value=(total_storages< typename LocalD::local_args_type, ArgType::index_type::value >::value);
        };

#ifdef CXX11_ENABLED
        /** @brief method called in the Do methods of the functors.
            specialization for the expr_direct_access<accessor> placeholders
        */
        template <typename ArgType>
        GT_FUNCTION
        typename boost::mpl::at<typename local_domain_t::esf_args, typename ArgType::type::index_type>::type::value_type& RESTRICT
        operator()(expr_direct_access<ArgType > const& arg) const {
            return get_value(arg, (*m_data_pointer)[current_storage<(ArgType::type::index_type::value==0), local_domain_t, typename ArgType::type >::value]);
        }
#endif

        /** @brief method called in the Do methods of the functors.
            specialization for the accessor placeholders

            this method is enabled only if the current placeholder dimension does not exceed the number of space dimensions of the storage class.
            I.e., if we are dealing with storages, not with storage lists or data fields (see concepts page for definitions)
        */
        template <typename ArgType>
        GT_FUNCTION
        typename boost::enable_if<
            typename boost::mpl::bool_< (ArgType::type::n_dim <=
                                         boost::mpl::at<
                                         typename local_domain_t::esf_args
                                         , typename ArgType::type::index_type>::type::storage_type::space_dimensions)>::type
                                        , typename boost::mpl::at<typename local_domain_t::esf_args
                                                                  , typename ArgType::type::index_type>::type::value_type
                                        >::type& RESTRICT
        operator()(ArgType const& arg) const {

            return get_value(arg, (*m_data_pointer)[current_storage<(ArgType::index_type::value==0)
                                                    , local_domain_t, typename ArgType::type >::value]);
        }

        /** @brief method called in the Do methods of the functors.
            Specialization for the accessor placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)

            this method is enabled only if the current placeholder dimension exceeds the number of space dimensions of the storage class.
            I.e., if we are dealing with  storage lists or data fields (see concepts page for definitions).
            TODO: This and the above version will be eventually merged.
        */
        template < typename ArgType>
        GT_FUNCTION
        typename boost::enable_if<
            typename boost::mpl::bool_<(ArgType::type::n_dim >
            boost::mpl::at<
            typename local_domain_t::esf_args
            , typename ArgType::type::index_type>::type::storage_type::space_dimensions)>::type
            , typename boost::mpl::at<typename local_domain_t::esf_args
            , typename ArgType::type::index_type>::type::value_type
        >::type&  RESTRICT
        operator()(ArgType const& arg) const;


#if defined(CXX11_ENABLED)
#if !defined(__CUDACC__)
        /** @brief method called in the Do methods of the functors.

            Specialization for the offset_tuple placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)*/
        template < typename ArgType, typename ... Pairs>
        GT_FUNCTION
        typename boost::mpl::at<typename local_domain_t::esf_args
                                , typename ArgType::index_type>::type::value_type& RESTRICT
        operator()(accessor_mixed<ArgType, Pairs ... > const& arg) const;

#endif //ifndef __CUDACC__

#endif

#ifdef CXX11_ENABLED
        /** @brief method called in the Do methods of the functors.

            specialization for the expr_direct_access<ArgType> placeholders (high level syntax: '@plch').
            Allows direct access to the storage by only using the offsets
        */
        template <typename ArgType, typename StoragePointer>
        GT_FUNCTION
        typename boost::mpl::at<typename local_domain_t::esf_args, typename ArgType::index_type>::type::value_type& RESTRICT
        get_value (expr_direct_access<ArgType> const& arg, StoragePointer & RESTRICT storage_pointer) const;


        /** @brief method called in the Do methods of the functors. */
        template <typename ... Arguments, template<typename ... Args> class Expression >
        GT_FUNCTION
        auto operator() (Expression<Arguments ... > const& arg) const ->decltype(evaluation::value(*this, arg)) {
            //arg.to_string();
            return evaluation::value((*this), arg);
        }

        /** @brief method called in the Do methods of the functors.
            partial specializations for double (or float)*/
        template <typename Arg, template<typename Arg1, typename Arg2> class Expression, typename FloatType
                  , typename boost::enable_if<typename boost::is_floating_point<FloatType>::type, int >::type=0 >
        GT_FUNCTION
        auto operator() (Expression<Arg, FloatType> const& arg) const ->decltype(evaluation::value_scalar(*this, arg)) {
            return evaluation::value_scalar((*this), arg);
        }

        /** @brief method called in the Do methods of the functors.
            partial specializations for int. Here we do not use the typedef int_t, because otherwise the interface would be polluted with casting
            (the user would have to cast all the numbers (-1, 0, 1, 2 .... ) to int_t before using them in the expression)*/
        // template <typename Arg, int Arg2, template<typename Arg1, int a> class Expression >
        template <typename Arg, template<typename Arg1, typename Arg2> class Expression, typename IntType
                  , typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0 >
        GT_FUNCTION
        auto operator() (Expression<Arg, IntType> const& arg) const ->decltype(evaluation::value_int((*this), arg)) {
            return evaluation::value_int((*this), arg);
        }

        template <typename Arg, template<typename Arg1, int Arg2> class Expression
                  , /*typename IntType, typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0*/int exponent >
        GT_FUNCTION
        auto operator() (Expression<Arg, exponent> const& arg) const ->decltype(evaluation::value_int((*this), arg)) {
            return evaluation::value_int((*this), arg);
        }

#endif // CXX11_ENABLED

    };

    /**@brief class handling the computation of the */
    template<typename IterateDomainImpl>
    struct positional_iterate_domain : public iterate_domain<IterateDomainImpl>
    {
        typedef iterate_domain<IterateDomainImpl> base_t;
        typedef typename base_t::local_domain_t local_domain_t;

#ifdef CXX11_ENABLED
        using iterate_domain<IterateDomainImpl>::iterate_domain;
#else
        GT_FUNCTION
        positional_iterate_domain(local_domain_t const& local_domain) : base_t(local_domain) {}
#endif

        /**@brief method for incrementing the index when moving forward along the k direction */
        template <ushort_t Coordinate, typename Execution>
        GT_FUNCTION
        void increment()
        {
            if (Coordinate==0) {
                m_i += Execution::value;
            }
            if (Coordinate==1) {
                m_j += Execution::value;
            }
            if( Coordinate==2)
                m_k += Execution::value;
            base_t::template increment<Coordinate, Execution>();
        }

        /**@brief method for incrementing the index when moving forward along the k direction */
        template <ushort_t Coordinate>
        GT_FUNCTION
        void increment(const uint_t steps_)
        {
            if (Coordinate==0) {
                m_i+=steps_;
            }
            if (Coordinate==1) {
                m_j+=steps_;
            }
            //TODOCOSUNA what with the increment of k? Is it GONE?
            base_t::template increment<Coordinate>(steps_);
            if( Coordinate==2)
                m_k += steps_;
        }

        template <ushort_t Coordinate>
        GT_FUNCTION
        void initialize(uint_t const& index=0, uint_t const& block=0)
        {
            if (Coordinate==0) {
                m_i = index;
            }
            if (Coordinate==1) {
                m_j = index;
            }
            base_t::template initialize<Coordinate>(index, block);
        }

        GT_FUNCTION
        uint_t i() const { return m_i;}

        GT_FUNCTION
        uint_t j() const { return m_j;}

        GT_FUNCTION
        uint_t k() const { return m_k;}

    private:
        uint_t m_i, m_j, m_k;
    };


//    ################## IMPLEMENTATION ##############################


    /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
       \param arg placeholder containing the storage ID and the offsets
       \param storage_pointer pointer to the first element of the specific data field used
    */
    template<typename IterateDomainImpl>
    template <typename ArgType, typename StoragePointer>
    GT_FUNCTION
    typename boost::mpl::at<typename iterate_domain_impl_local_domain<IterateDomainImpl>::type::esf_args, typename ArgType::index_type>::type::value_type& RESTRICT
    iterate_domain<IterateDomainImpl>::get_value(ArgType const& arg , StoragePointer & RESTRICT storage_pointer) const {


#ifdef CXX11_ENABLED
        using storage_type = typename std::remove_reference<decltype(*boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))>::type;
#else
        typedef typename boost::remove_reference<BOOST_TYPEOF( (*boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)) )>::type storage_type;
#endif
        typename storage_type::value_type * RESTRICT real_storage_pointer=static_cast<typename storage_type::value_type*>(storage_pointer);

        //the following assert fails when an out of bound access is observed, i.e. either one of
        //i+offset_i or j+offset_j or k+offset_k is too large.
        //Most probably this is due to you specifying a positive offset which is larger than expected,
        //or maybe you did a mistake when specifying the ranges in the placehoders definition
        assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->size() >  m_index[ArgType::index_type::value]
               +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
               ->_index(m_strides->template get<ArgType::index_type::value>(), arg)
            );

        //the following assert fails when an out of bound access is observed,
        //i.e. when some offset is negative and either one of
        //i+offset_i or j+offset_j or k+offset_k is too small.
        //Most probably this is due to you specifying a negative offset which is
        //smaller than expected, or maybe you did a mistake when specifying the ranges
        //in the placehoders definition.
        // If you are running a parallel simulation another common reason for this to happen is
        // the definition of an halo region which is too small in one direction
        // std::cout<<"Storage Index: "<<ArgType::index_type::value<<" + "<<(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))->_index(arg.template n<ArgType::n_dim>())<<std::endl;
        assert( (int_t)(m_index[ArgType::index_type::value])
                +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                ->_index(m_strides->template get<ArgType::index_type::value>(), arg)
                >= 0);

        return *(real_storage_pointer
                 +(m_index[ArgType::index_type::value])
                 +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                 //here we suppose for the moment that ArgType::index_types are ordered like the LocalDomain::esf_args mpl vector
                 ->_index(m_strides->template get<ArgType::index_type::value>(), arg)
            );
    }


    /** @brief method called in the Do methods of the functors.
        Specialization for the offset_tuple placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)*/
    template<typename IterateDomainImpl>
    template < typename ArgType>
    GT_FUNCTION
    typename boost::enable_if<
        typename boost::mpl::bool_<(ArgType::type::n_dim > boost::mpl::at<typename iterate_domain_impl_local_domain<IterateDomainImpl>::type::esf_args, typename ArgType::type::index_type>::type::storage_type::space_dimensions)>::type
        , typename boost::mpl::at<typename iterate_domain_impl_local_domain<IterateDomainImpl>::type::esf_args, typename ArgType::type::index_type>::type::value_type >::type&  RESTRICT
    iterate_domain<IterateDomainImpl>::operator()(ArgType const& arg) const {

#ifdef CXX11_ENABLED
        using storage_type = typename std::remove_reference<decltype(*boost::fusion::at<typename ArgType::type::index_type>(local_domain.local_args))>::type;
#else
        typedef typename boost::remove_reference<BOOST_TYPEOF( (*boost::fusion::at<typename ArgType::type::index_type>(local_domain.local_args)) )>::type storage_type;
#endif
        //if the following assertion fails you have specified a dimension for the extended storage
        //which does not correspond to the size of the extended placeholder for that storage
        GRIDTOOLS_STATIC_ASSERT(storage_type::space_dimensions+2/*max. extra dimensions*/>=ArgType::type::n_dim, "the dimension of the accessor exceeds the data field dimension");


        //for the moment the extra dimensionality of the storage is limited to max 2
        //(3 space dim + 2 extra= 5, which gives n_dim==4)
        GRIDTOOLS_STATIC_ASSERT(N_DATA_POINTERS>0, "the total number of snapshots must be larger than 0 in each functor");
        GRIDTOOLS_STATIC_ASSERT(ArgType::type::n_dim <= ArgType::type::n_dim, "access out of bound in the storage placeholder (accessor). increase the number of dimensions when defining the placeholder.");

        GRIDTOOLS_STATIC_ASSERT((storage_type::traits::n_fields%storage_type::traits::n_width==0), "You specified a non-rectangular field: if you need to use a non-rectangular field the constexpr version of the accessors have to be used (so that the current position in the field is computed at compile time). This is achieved by using, e.g., instead of \n\n eval(field(Dimension<5>(2))); \n\n the following expression: \n\n typedef alias<field, Dimension<5> >::set<2> z_field; \n eval(z_field()); \n");

            //dimension/snapshot offsets must be non negative
        assert(arg.template get<0>()>=0);
        assert( (ArgType::type::n_dim <= storage_type::space_dimensions+1) ||
                (arg.template get<1>()>=0) );
             // std::cout<<" offsets: "<<arg.template get<0>()<<" , "<<arg.template get<1>()<<" , "<<arg.template get<2>()<<" , "<<std::endl;

        return get_value(arg,
                         (*m_data_pointer)[
                             (
                                 ArgType::type::n_dim <= storage_type::space_dimensions+1 ? // static if
                                 arg.template get<0>() //offset for the current dimension
                                 :
                                 arg.template get<1>() //offset for the current snapshot
                                 //limitation to "rectangular" vector fields for non-static fields dimensions
                                 +  arg.template get<0>() //select the dimension
                                 * storage_type::traits::n_width  //stride of the current dimension inside the vector of storages
                                 )
                                     //+ the offset of the other extra dimension
                             + current_storage<(ArgType::type::index_type::value==0), local_domain_t, typename ArgType::type>::value
                             ]);
    }

#if defined(CXX11_ENABLED) && !defined( __CUDACC__ )
    /** @brief method called in the Do methods of the functors.

        Specialization for the offset_tuple placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)*/
    template <typename IterateDomainImpl>
    template < typename ArgType, typename ... Pairs>
    GT_FUNCTION
    typename boost::mpl::at<typename iterate_domain_impl_local_domain<IterateDomainImpl>::type::esf_args
                            , typename ArgType::index_type>::type::value_type& RESTRICT
    iterate_domain<IterateDomainImpl>::operator()(accessor_mixed<ArgType, Pairs ... > const& arg) const{

        typedef accessor_mixed<ArgType, Pairs ... > accessor_mixed_t;
        using storage_type = typename std::remove_reference<decltype(*boost::fusion::at<typename ArgType::type::index_type>(local_domain.local_args))>::type;

        //if the following assertion fails you have specified a dimension for the extended storage
        //which does not correspond to the size of the extended placeholder for that storage
        /* BOOST_STATIC_ASSERT(storage_type::n_fields==ArgType::n_dim); */

        //for the moment the extra dimensionality of the storage is limited to max 2
        //(3 space dim + 2 extra= 5, which gives n_dim==4)
        GRIDTOOLS_STATIC_ASSERT(N_DATA_POINTERS>0, "the total number of snapshots must be larger than 0 in each functor");
        GRIDTOOLS_STATIC_ASSERT(ArgType::type::n_dim <= ArgType::type::n_dim, "access out of bound in the storage placeholder (accessor). increase the number of dimensions when defining the placeholder.");
        GRIDTOOLS_STATIC_ASSERT(accessor_mixed_t::template get_constexpr<0>()>=0,
                                "offset specified for the dimension corresponding to the number of snapshots must be non negative");
        GRIDTOOLS_STATIC_ASSERT( (ArgType::type::n_dim <= storage_type::space_dimensions+1) ||
                                     (accessor_mixed_t::template get_constexpr<1>()>=0),
                                 "offset specified for the dimension corresponding to the number of field components must be non negative");

        return get_value(arg,
                         (*m_data_pointer)[ //static if
                             //TODO: re implement offsets in accessor which can be or not constexpr (not in a vector)
               (
                   ArgType::type::n_dim <= storage_type::space_dimensions+1 ? // static if
                   accessor_mixed_t::template get_constexpr<0>() //offset for the current snapshot
                   :
                   accessor_mixed_t::template get_constexpr<1>() //offset for the current snapshot
                   //hypotheses : storage offsets are known at compile-time
                   + compute_storage_offset< typename storage_type::traits
                   , accessor_mixed_t::template get_constexpr<0>()
                   , storage_type::traits::n_dimensions-1 >::value //stride of the current dimension inside the vector of storages
                   )//+ the offset of the other extra dimension
               + current_storage<(ArgType::index_type::value==0), local_domain_t, typename ArgType::type>::value
                             ]);
    }


    /** @brief method called in the Do methods of the functors.

        specialization for the expr_direct_access<ArgType> placeholders (high level syntax: '@plch').
        Allows direct access to the storage by only using the offsets
    */
    template <typename IterateDomainImpl>
    template <typename ArgType, typename StoragePointer>
    GT_FUNCTION
    typename boost::mpl::at<typename iterate_domain_impl_local_domain<IterateDomainImpl>::type::esf_args, typename ArgType::index_type>::type::value_type& RESTRICT
    iterate_domain<IterateDomainImpl>::get_value (expr_direct_access<ArgType> const& arg, StoragePointer & RESTRICT storage_pointer) const {

        assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->size() >  (boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
               ->_index(m_strides->template get<ArgType::index_type::value>(), arg.first_operand));

        assert((boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
               ->_index(m_strides->template get<ArgType::index_type::value>(), arg.first_operand) >= 0);
        GRIDTOOLS_STATIC_ASSERT((
                                    ArgType::n_dim <= boost::mpl::at<
                                    typename local_domain_t::esf_args,
                                    typename ArgType::index_type
                                    >::type::storage_type::space_dimensions),
                                "the dimension of the storage placeholder (accessor) is lower than the number of space dimensions of the corresponding storage. Either increase the number of storage dimensions, or decrease the dimension of the accessor.");

        using storage_type = typename std::remove_reference<decltype(*boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))>::type;

        typename storage_type::value_type * RESTRICT real_storage_pointer=static_cast<typename storage_type::value_type*>(storage_pointer);

        return *(real_storage_pointer
                 +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                 ->_index(m_strides->template get<ArgType::index_type::value>(), arg.first_operand));
    }
#endif //ifndef CXX11_ENABLED

} // namespace gridtools
