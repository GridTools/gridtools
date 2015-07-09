#pragma once
#include "storage_list.hpp"

namespace gridtools{
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
    /** @brief traits class defining some useful compile-time counters
     */
    template < typename First, typename  ...  StorageExtended>
    struct dimension_extension_traits
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits<StorageExtended  ...  >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits<StorageExtended  ...  >::n_dimensions  +1 ;
        //the current field extension
        //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
        typedef storage_list<First, n_fields-1>  type;
        // typedef First type;
        typedef dimension_extension_traits<StorageExtended ... > super;
    };


    template<typename T>
    struct get_fields{
        using type = static_int<T::n_fields>;
    };

    template<typename T>
    struct get_value{
        using type = static_int<T::value>;
    };

    template<typename Storage, uint_t Id, uint_t IdMax>
    struct compute_storage_offset{

        GRIDTOOLS_STATIC_ASSERT(IdMax>=Id && Id>=0, "Library internal error");
        typedef typename boost::mpl::eval_if_c<IdMax-Id==0, get_fields<typename Storage::super> , get_value<compute_storage_offset<typename Storage::super, Id+1, IdMax> > >::type type;
        static const uint_t value=type::value;
    };

#endif

    /**@brief fallback in case the snapshot we try to access exceeds the width diemnsion assigned to a discrete scalar field*/
    struct dimension_extension_null{
        static const ushort_t n_fields=0;
        static const short_t n_width=0;
        static const ushort_t n_dimensions=0;
        typedef struct error_index_too_large1{} type;
        typedef struct error_index_too_large2{} super;
    };

/**@brief template specialization at the end of the recustion.*/
    template < typename First>
    struct dimension_extension_traits
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
    <First>
#endif
    {
        static const ushort_t n_fields=First::n_width;
        static const short_t n_width=First::n_width;
        static const ushort_t n_dimensions= 1 ;
        typedef First type;
        typedef dimension_extension_null super;
    };

#if !defined(CXX11_ENABLED) || defined(__CUDACC__)// big code repetition

    /** @brief non-C++11 crap
     */
    template < typename First, typename Second>
    struct dimension_extension_traits2// : public dimension_extension_traits<StorageExtended ... >
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits< Second >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits< Second>::n_dimensions  +1 ;
        //the current field extension
   //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
   typedef storage_list<First, n_fields-1>  type;
   // typedef First type;
   typedef dimension_extension_traits< Second> super;
    };

    template < typename First, typename Second, typename Third>
    struct dimension_extension_traits3// : public dimension_extension_traits<StorageExtended ... >
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits2< Second, Third >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits2< Second, Third >::n_dimensions  +1 ;
        //the current field extension
   //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
   typedef storage_list<First, n_fields-1>  type;
   // typedef First type;
   typedef dimension_extension_traits2< Second, Third > super;
    };
#endif //CXX11_ENABLED


#if defined( CXX11_ENABLED ) && !defined( __CUDACC__ )
    /**@brief implements the discretized field structure

       It is a collection of arbitrary length \ref gridtools::storage_list "storage lists".

       \include storage.dox

    */
    template <typename First,  typename  ...  StorageExtended>
    struct data_field : public dimension_extension_traits<First, StorageExtended ... >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
    {
        typedef data_field<First, StorageExtended...> type;
        typedef typename dimension_extension_traits<First, StorageExtended ... >::type super;
        typedef dimension_extension_traits<First, StorageExtended ...  > traits;
        typedef typename super::pointer_type pointer_type;
        typedef typename  super::basic_type basic_type;
        typedef typename super::original_storage original_storage;
        static const short_t n_width=sizeof...(StorageExtended)+1;

        /**@brief constructor given the space boundaries*/
        template<typename ... UIntTypes>
        data_field(  UIntTypes const& ... args )
            : super(args...)
            {
       }
#else

        template <typename First, typename Second, typename Third>
        struct data_field : public dimension_extension_traits3<First, Second, Third >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
        {
            typedef data_field<First, Second, Third> type;
            typedef typename dimension_extension_traits3<First, Second, Third >::type super;
            typedef dimension_extension_traits3<First, Second, Third > traits;
            typedef typename super::pointer_type pointer_type;
            typedef typename  super::basic_type basic_type;
            typedef typename super::original_storage original_storage;
            static const short_t n_width=3;

            /**@brief constructor given the space boundaries*/
            data_field(  uint_t const& d1, uint_t const& d2, uint_t const& d3 )
                : super(d1, d2, d3)
            {
       }

#endif

   /**@brief default constructor*/
        data_field(): super(){}

   /**@brief device copy constructor*/
        template <typename T>
        __device__
        data_field( T const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field(){
        }

        using super::setup;

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
           \tparam dimension specifies which field dimension we want to access
        */
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

            /*If the following assertion fails your field dimension is smaller than the dimension you are trying to access*/
            BOOST_STATIC_ASSERT(n_width>dimension);
            /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width<=traits::n_fields);
            uint_t const indexFrom=_impl::access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<n_width-dimension-1, traits>::type::n_fields;
            super::push_front(field, indexFrom, indexTo);
        }

   /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
       for (uint_t i=0; i<super::size(); ++i)
           field[i]=value;
       push_front<dimension>(field);
   }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot>
#endif
   void set( pointer_type& field)
       {
           super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param val the initializer value
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot>
#endif
   void set(/* pointer_type& field,*/ typename super::value_type const& val)
       {
           for (uint_t i=0; i<super::size(); ++i)
               (super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot])[i]=val;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function
      TODO: this should be merged with the boundary conditions code (repetition)

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param lambda the initializer function
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot  >
#endif
   void set( typename super::value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&))
       {
           for (uint_t i=0; i<this->m_dims[0]; ++i)
               for (uint_t j=0; j<this->m_dims[1]; ++j)
                   for (uint_t k=0; k<this->m_dims[2]; ++k)
                       (super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot])
                           [super::_index(super::strides(), i,j,k)]=
                           lambda(i, j, k);
       }


   /**@biref gets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim  , short_t snapshot  >
#endif
   pointer_type& get( )
       {
           return super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot];
       }


   /**@biref sets the given storage as the nth snapshot of a specific field dimension, at the specified coordinates


        /**@biref gets a given value as the given field i,j,k coordinates

           \tparam field_dim the given field dimenisons
           \tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           \param i index in the horizontal direction
           \param j index in the horizontal direction
           \param k index in the vertical direction
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot  >
#endif
        typename super::value_type& get_value( uint_t const& i, uint_t const& j, uint_t const& k )
      {
                    return get<field_dim, snapshot>()[super::_index(super::strides(),i,j,k)];
      }

        /**@biref ODE advancing for a single dimension

           it advances the supposed finite difference scheme of one step for a specific field dimension
           \tparam dimension the dimension to be advanced
           \param offset the number of steps to advance
        */
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=_impl::access<dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

        /**@biref ODE advancing for all dimension

           shifts the rings of solutions of one position,
           it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all(){
            _impl::advance_recursive<n_width>::apply(const_cast<data_field*>(this));
        }

    };

#if !defined( CXX11_ENABLED ) || defined ( __CUDACC__ )
        template <typename First, typename Second>
        struct data_field2 : public dimension_extension_traits2<First, Second >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
        {
            typedef data_field2<First, Second> type;
            typedef typename dimension_extension_traits2<First, Second >::type super;
            typedef dimension_extension_traits2<First, Second > traits;
            typedef typename super::pointer_type pointer_type;
            typedef typename  super::basic_type basic_type;
            typedef typename super::original_storage original_storage;
            static const short_t n_width=2;

            /**@brief constructor given the space boundaries*/
            data_field2(  uint_t const& d1, uint_t const& d2, uint_t const& d3 )
                : super(d1, d2, d3)
            {
       }

   /**@brief device copy constructor*/
        template <typename T>
        __device__
        data_field2( T const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field2(){
   }

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
      \tparam dimension specifies which field dimension we want to access
        */
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

       /*If the following assertion fails your field dimension is smaller than the dimension you are trying to access*/
            BOOST_STATIC_ASSERT(n_width>dimension);
       /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width<=traits::n_fields);
            uint_t const indexFrom=_impl::access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<n_width-dimension-1, traits>::type::n_fields;
       super::push_front(field, indexFrom, indexTo);
        }

   /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
       for (uint_t i=0; i<super::size(); ++i)
           field[i]=value;
       push_front<dimension>(field);
   }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field)
       {
      super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param val the initializer value
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field, typename super::value_type const& val)
       {
      for (uint_t i=0; i<super::size(); ++i)
          field[i]=val;
      set<field_dim, snapshot>(field);
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function
      TODO: this should be merged with the boundary conditions code (repetition)

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param lambda the initializer function
        */
   template<short_t field_dim, short_t snapshot  >
   void set( pointer_type& field, typename super::value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&))
       {
      for (uint_t i=0; i<this->m_dims[0]; ++i)
          for (uint_t j=0; j<this->m_dims[1]; ++j)
         for (uint_t k=0; k<this->m_dims[2]; ++k)
             (field)[super::_index(super::strides(), i,j,k)]=lambda(i, j, k);
      set<field_dim, snapshot>(field);
       }


   /**@biref gets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim  , short_t snapshot  >
   pointer_type& get( )
       {
      return super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot];
       }


   /**@biref sets the given storage as the nth snapshot of a specific field dimension, at the specified coordinates

           If on the device, it calls the API to set the memory on the device
      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param value the value to be set
        */

        /**@biref gets a given value as the given field i,j,k coordinates

           \tparam field_dim the given field dimenisons
           \tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           \param i index in the horizontal direction
           \param j index in the horizontal direction
           \param k index in the vertical direction
        */
   template<short_t field_dim, short_t snapshot  >
        typename super::value_type& get_value( uint_t const& i, uint_t const& j, uint_t const& k )
      {
                    return get<field_dim, snapshot>()[super::_index(super::strides(),i,j,k)];
      }

   /**@biref ODE advancing for a single dimension

      it advances the supposed finite difference scheme of one step for a specific field dimension
      \tparam dimension the dimension to be advanced
      \param offset the number of steps to advance
        */
        template<uint_t dimension>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=_impl::access<dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

   /**@biref ODE advancing for all dimension

      shifts the rings of solutions of one position,
      it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all(){
       _impl::advance_recursive<n_width>::apply(const_cast<data_field2*>(this));
        }

#ifdef NDEBUG
    private:
        //for stdcout purposes
   data_field2();
#else
        data_field2(){}
#endif
    };

        template <typename First>
        struct data_field1 : public dimension_extension_traits<First >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
        {
            typedef data_field1<First> type;
            typedef typename dimension_extension_traits<First >::type super;
            typedef dimension_extension_traits<First > traits;
            typedef typename super::pointer_type pointer_type;
            typedef typename  super::basic_type basic_type;
            typedef typename super::original_storage original_storage;
            static const short_t n_width=1;

            /**@brief constructor given the space boundaries*/
            data_field1(  uint_t const& d1, uint_t const& d2, uint_t const& d3 )
                : super(d1, d2, d3)
            {
       }

   /**@brief device copy constructor*/
        template <typename T>
        __device__
        data_field1( T const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field1(){
   }

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
      \tparam dimension specifies which field dimension we want to access
        */
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

       /*If the following assertion fails your field dimension is smaller than the dimension you are trying to access*/
            BOOST_STATIC_ASSERT(n_width>dimension);
       /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width<=traits::n_fields);
            uint_t const indexFrom=_impl::access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<n_width-dimension-1, traits>::type::n_fields;
       super::push_front(field, indexFrom, indexTo);
        }

   /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
       for (uint_t i=0; i<super::size(); ++i)
           field[i]=value;
       push_front<dimension>(field);
   }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field)
       {
      super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param val the initializer value
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field, typename super::value_type const& val)
       {
      for (uint_t i=0; i<super::size(); ++i)
          field[i]=val;
      set<field_dim, snapshot>(field);
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function
      TODO: this should be merged with the boundary conditions code (repetition)

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param lambda the initializer function
        */
   template<short_t field_dim, short_t snapshot  >
   void set( pointer_type& field, typename super::value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&))
       {
      for (uint_t i=0; i<this->m_dims[0]; ++i)
          for (uint_t j=0; j<this->m_dims[1]; ++j)
         for (uint_t k=0; k<this->m_dims[2]; ++k)
             (field)[super::_index(super::strides(), i,j,k)]=lambda(i, j, k);
      set<field_dim, snapshot>(field);
       }


   /**@biref gets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim  , short_t snapshot  >
   pointer_type& get( )
       {
      return super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot];
       }


   /**@biref sets the given storage as the nth snapshot of a specific field dimension, at the specified coordinates

           If on the device, it calls the API to set the memory on the device
      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param value the value to be set
        */

        /**@biref gets a given value as the given field i,j,k coordinates

           \tparam field_dim the given field dimenisons
           \tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           \param i index in the horizontal direction
           \param j index in the horizontal direction
           \param k index in the vertical direction
        */
   template<short_t field_dim, short_t snapshot  >
        typename super::value_type& get_value( uint_t const& i, uint_t const& j, uint_t const& k )
      {
                    return get<field_dim, snapshot>()[super::_index(super::strides(),i,j,k)];
      }

   /**@biref ODE advancing for a single dimension

      it advances the supposed finite difference scheme of one step for a specific field dimension
      \tparam dimension the dimension to be advanced
      \param offset the number of steps to advance
        */
        template<uint_t dimension>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=_impl::access<dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

   /**@biref ODE advancing for all dimension

      shifts the rings of solutions of one position,
      it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all(){
       _impl::advance_recursive<n_width>::apply(const_cast<data_field1*>(this));
        }

#ifdef NDEBUG
    private:
        //for stdcout purposes
   data_field1();
#else
        data_field1(){}
#endif
    };
#endif


#if defined(CXX11_ENABLED) && !defined( __CUDACC__ )
    template <typename F, typename ... T>
    std::ostream& operator<<(std::ostream &s, data_field< F, T... > const &) {
        return s << "field storage" ;
    }
#endif

}//namespace gridtools
