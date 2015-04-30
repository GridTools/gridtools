#pragma once

#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include <boost/fusion/include/size.hpp>
#include "expressions.h"
#include "arg_type.h"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools{
    /**
       @brief struct to allocate recursively all the strides with the proper dimension

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageList typelist of the storages
    */
    template<uint_t ID, typename StorageList>
    struct strides_cached : public strides_cached<ID-1, StorageList> {
        typedef typename  boost::mpl::at_c<StorageList, ID>::type::storage_type storage_type;
        typedef strides_cached<ID-1, StorageList> super;

        /**@brief constructor, doing nothing more than allocating the space*/
        GT_FUNCTION
            strides_cached():super(){
            GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<StorageList>::value > ID, "Library internal error: strides index exceeds the number of storages")
                }

        template<short_t Idx>
            GT_FUNCTION
            uint_t * RESTRICT get() {
            return ((Idx==ID-1)? &m_data[0] : (super::template get<Idx>()));
        }

    private:
        uint_t m_data[storage_type::space_dimensions-1];
    };


    /**specialization to stop the recursion*/
    template<typename storage_list>
    struct strides_cached<0, storage_list>  {
        typedef typename boost::mpl::at_c<storage_list, 0>::type::storage_type storage_type;

        GT_FUNCTION
        strides_cached(){}

        template<short_t Idx>
        GT_FUNCTION
        uint_t * RESTRICT get() {//stop recursion
            //GRIDTOOLS_STATIC_ASSERT(Idx==0, "Internal library error: Index exceeding the storage_list dimension.")
            return &m_data[0];
        }

    private:
        uint_t m_data[storage_type::space_dimensions-1];
    };


    //defines how many threads participate to the (shared) memory initialization
#define BLOCK_SIZE 32

    /**@brief recursively assigning the 'raw' data pointers to an input data pointers array (i.e. the m_data_pointers array).

       The 'raw' datas are the one or more data fields contained in each of the storage classes used by the current user function.
       \tparam Number the recursion index, identifying the current snapshot whithin a storage
       \tparam Offset an index identifying the starting position in the data pointers array of the portion corresponding to the given storage
       \tparam BackendType the type of backend
       \tparam StrategyType the tyupe of strategy

       To clarify the meaning of the two template indices, supposing that we have a 'rectangular' vector field, NxM, where N is the constant number of
       snapshots per storage, while M is the number of storages. Then 'Number' would be an index between 0 and N, while Offset would have the form n*M, where
       0<n<N is the index of the previous storage.
    */
    template<uint_t Number, uint_t Offset, typename BackendType, typename StorageType>
    struct assign_raw_data{
        typedef StorageType storage_type;
        static const uint_t Id=(Number+Offset)%BLOCK_SIZE;

        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const& RESTRICT r, int EU_id_i, int EU_id_j,
                           typename boost::enable_if_c<is_host_tmp_storage<Right>::value>::type* = 0)
            {
                BackendType::template once_per_block<Id>::assign(l[Offset+Number],r->field_offset(Number,EU_id_i, EU_id_j));
                assign_raw_data<Number-1, Offset, BackendType, storage_type>::assign(l, r, EU_id_i, EU_id_j);
            }

        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const& RESTRICT r, int EU_id_i, int EU_id_j,
                           typename boost::disable_if_c<is_host_tmp_storage<Right>::value>::type* = 0)
            {
                BackendType::template once_per_block<Id>::assign(l[Offset+Number], r->fields()[Number].get());
                assign_raw_data<Number-1, Offset, BackendType, storage_type>::assign(l, r, EU_id_i, EU_id_j);
            }

    };

    /**@brief stopping the recursion*/
    template<uint_t Offset, typename BackendType, typename StorageType>
    struct assign_raw_data<0, Offset, BackendType, StorageType>{
        typedef StorageType storage_type;
        static const uint_t Id=(Offset)%BLOCK_SIZE;

        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const& RESTRICT r, int EU_id_i, int EU_id_j,
                           typename boost::enable_if_c<is_host_tmp_storage<Right>::value>::type* = 0)
            {
                BackendType:: template once_per_block<Id>::assign(l[Offset],r->fields_offset(0,EU_id_i, EU_id_j));
            }

        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const& RESTRICT r, int EU_id_i, int EU_id_j,
                           typename boost::disable_if_c<is_host_tmp_storage<Right>::value>::type* = 0)
            {
                BackendType:: template once_per_block<Id>::assign(l[Offset],r->fields()[0].get());
            }
    };

    /**@brief this struct counts the total number of data fields which are neceassary for this functor (i.e. number of storage instances times number of fields per storage)
       TODO code repetition in the _traits class
    */
    template <typename StoragesVector, int_t index>
    struct total_storages{
        //the index must not exceed the number of storages
        GRIDTOOLS_STATIC_ASSERT(index<boost::mpl::size<StoragesVector>::type::value, "the index must not exceed the number of storages")
        static const uint_t count=total_storages<StoragesVector, index-1>::count+
            boost::remove_pointer<typename boost::remove_reference<typename boost::mpl::at_c<StoragesVector, index >::type>::type>
            ::type::field_dimensions;
    };

    /**@brief partial specialization to stop the recursion*/
    template <typename StoragesVector>
    struct total_storages<StoragesVector, 0 >{
        static const uint_t count=boost::remove_pointer<typename boost::remove_reference<typename  boost::mpl::at_c<StoragesVector, 0 >::type>::type>::type::field_dimensions;
    };

    /**@brief incrementing all the storage pointers to the m_data_pointers array

       \tparam ID identifier for the current storage (recursion index)
       \tparam Coordinate direction along which the increment takes place
       \tparam Execution policy determining how the increment is done (e.g. increment/decrement)
    */
    template<uint_t ID, uint_t Coordinate, enumtype::execution Execution>
    struct increment_index {

        /**@brief does the actual assignment

           This method is responsible of incrementing the index for the memory access at
           the location (i,j,k) incremented/decremented by 1 along the 'Coordinate' direction. Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           The actual increment computation is delegated to the storage classes, the reason being that the implementation may depend on the storage type
           (e.g. whether the storage is temporary, partiitoned into blocks, ...)
        */
        template<typename Storage, typename Strides>
        GT_FUNCTION
        static void assign(Storage const& r_, uint_t* RESTRICT index_, Strides &  RESTRICT strides_){
            boost::fusion::at_c<ID>(r_)->template increment<Coordinate, Execution>(&index_[ID], strides_.template get<ID>());
            increment_index<ID-1, Coordinate, Execution>::assign(r_,index_,strides_);
        }

        /** @brief method for computing the index rof the memory access at a specific (i,j,k) location incremented/decremented by 'increment_' in direction 'Coordinate'.
         */
        template<typename Storage, typename Strides>
        GT_FUNCTION
        static void assign(Storage const& r_, uint const& increment_, uint_t* RESTRICT index_, Strides &  RESTRICT strides_){
            boost::fusion::at_c<ID>(r_)->template increment<Coordinate, Execution>(increment_,&index_[ID], strides_.template get<ID>());
            increment_index<ID-1, Coordinate, Execution>::assign(r_,increment_,index_,strides_);
        }

    };


    /**usual specialization to stop the recursion*/
    template<uint_t Coordinate, enumtype::execution Execution>
    struct increment_index<0, Coordinate, Execution>{

        template<typename Storage
                 , typename Strides
                 >
        GT_FUNCTION
        static void assign( Storage const &  r_, uint_t* RESTRICT index_, Strides & RESTRICT strides_){
            boost::fusion::at_c<0>(r_)->template increment<Coordinate, Execution>(&index_[0], (strides_.template get<0>() )
                );
        }

        template<typename Storage, typename Strides>
        GT_FUNCTION
        static void assign(Storage const& r_, uint const& increment_, uint_t* RESTRICT index_, Strides &  RESTRICT strides_){
            boost::fusion::at_c<0>(r_)->template increment<Coordinate, Execution>(increment_,&index_[0], strides_.template get<0>());
        }

    };


    /**@brief assigning all the storage pointers to the m_data_pointers array*/
    template<uint_t ID, uint_t Coordinate>
    struct advance_index {

        /**@brief does the actual assignment
           This method is responsible of computing the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.
        */
        template<typename Storage>
        GT_FUNCTION
        static void advance(Storage & r, uint_t id, uint_t block, uint_t* index){
            //if the following fails, the ID is larger than the number of storage types,
            //or the index was not properly initialized to 0,
            //or you know what you are doing (then comment out the assert)
            boost::fusion::at_c<ID>(r)->template increment<Coordinate>(id, block, &index[ID]);
            advance_index<ID-1, Coordinate>::advance(r,id,block,index);
        }
    };

    /**usual specialization to stop the recursion*/
    template<uint_t Coordinate>
        struct advance_index<0, Coordinate>{
        template<typename Storage>
        GT_FUNCTION
        static void advance( Storage & r, uint_t id, uint_t block, uint_t* index/* , ushort_t* lru */){

            boost::fusion::at_c<0>(r)->template increment<Coordinate>(id, block, &index[0]);
        }
    };


    /**@brief assigning all the storage pointers to the m_data_pointers array

       similar to the increment_index class, but assigns the indices, and it does not depend on the storage type
    */
    template<uint_t ID>
    struct set_index_recur{
        /**@brief does the actual assignment
           This method is responsible of assigning the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           This method given an array and an integer id assigns to the current component of the array the input integer.
        */
        template <typename Array>
        GT_FUNCTION
        static void set(uint_t const& id, Array& index){
            index[ID]=id;
            set_index_recur<ID-1>::set(id,index);
        }

        /**@brief does the actual assignment
           This method is responsible of assigning the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           This method given two arrays copies the IDth component of one into the other, i.e. recursively cpoies one array into the other.
        */
        template<typename Array>
        GT_FUNCTION
        static void set(Array const& index, Array& out){
            out[ID]=index[ID];
            set_index_recur<ID-1>::set(index, out);
        }
    };


    /**usual specialization to stop the recursion*/
    template<>
    struct set_index_recur<0>{

        template<typename Array>
        GT_FUNCTION
        static void set( uint_t const& id, Array& index/* , ushort_t* lru */){
            index[0]=id;
        }

        template<typename Array>
        GT_FUNCTION
        static void set(Array const& index, Array& out){
            out[0]=index[0];
        }
    };


    /**@brief initializing the indeces*/
    template<uint_t ID, uint_t Coordinate>
    struct initialize_index {

        /**@brief does the actual assignment
           This method is responsible of computing the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.
        */
        template<typename Storage, typename Strides>
        GT_FUNCTION
        static void assign(Storage const& RESTRICT r_, uint_t id_, uint_t block_, uint_t* RESTRICT index_, Strides &  RESTRICT strides_){

            boost::fusion::at_c<ID>(r_)->template initialize<Coordinate>(id_, block_, &index_[ID], strides_.template get<ID>());
            initialize_index<ID-1, Coordinate>::assign(r_, id_,block_,index_,strides_);
        }
    };

    /**usual specialization to stop the recursion*/
    template<uint_t Coordinate>
    struct initialize_index<0, Coordinate>{

        template<typename Storage
                 , typename Strides
                 >
        GT_FUNCTION
        static void assign( Storage const & RESTRICT r_, uint_t id_, uint_t block_, uint_t* RESTRICT index_, Strides & RESTRICT strides_){

            boost::fusion::at_c<0>(r_)->template initialize<Coordinate>(id_, block_, &index_[0], (strides_.template get<0>() )
                );
        }
    };

    /**@brief assigning all the storage pointers to the m_data_pointers array*/
    template<uint_t ID, typename BackendType>
    struct assign_storage{

        /**@brief does the actual assignment
           This method is responsible of copying the base pointers of the storages inside a local vector
           which is tipically instantiated on a fast local memory.

           The EU stands for ExecutionUnit (thich may be a thread or a group of
           threads. There are potentially two ids, one over i and one over j, since
           our execution model is parallel on (i,j). Defaulted to 1.
        */
        template<typename Left, typename Right>
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const & RESTRICT r, int EU_id_i, int EU_id_j){
#ifdef CXX11_ENABLED
            typedef typename std::remove_pointer
                <typename std::remove_reference<decltype(boost::fusion::at_c<ID>(r))>::type>::type storage_type;
#else
            typedef typename boost::remove_pointer
                <typename boost::remove_reference
                 <BOOST_TYPEOF(boost::fusion::at_c<ID>(r))
                  >::type
                 >::type storage_type;
#endif
            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID < boost::mpl::size<Right>::value, "the ID is larger than the number of storage types")

                assign_raw_data<storage_type::field_dimensions-1,
                                total_storages<Right, ID-1>::count,
                                BackendType,
                                storage_type>::
                assign(l, boost::fusion::at_c<ID>(r),
                       EU_id_i, EU_id_j);

            assign_storage<ID-1, BackendType>::assign(l, r, EU_id_i, EU_id_j); //tail recursion
        }
    };

    /**usual specialization to stop the recursion*/
    template<typename BackendType>
    struct assign_storage<0, BackendType>{

        template<typename Left, typename Right>
        GT_FUNCTION
        static void assign(Left & RESTRICT l, Right const & RESTRICT r, int EU_id_i, int EU_id_j){
#ifdef CXX11_ENABLED
            typedef typename std::remove_pointer< typename std::remove_reference<decltype(boost::fusion::at_c<0>(r))>::type>::type storage_type;
#else
            typedef typename boost::remove_pointer< typename boost::remove_reference<BOOST_TYPEOF(boost::fusion::at_c<0>(r))>::type>::type storage_type;
#endif
            // std::cout<<"ID is: "<<0<<"n_width is: "<< storage_type::n_width-1 << "current index is "<< 0 <<std::endl;
            assign_raw_data<storage_type::field_dimensions-1, 0, BackendType, storage_type>::
                assign(l, boost::fusion::at_c<0>(r), EU_id_i, EU_id_j);
        }
    };


    /**
       @brief assigning the storage strides to the m_strides array.

       This is the unrolling of the inner nested loop
    */
    template<uint_t Number, typename BackendType>
    struct assign_strides_rec{
        static const uint_t Id=(Number)%BLOCK_SIZE;
        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left* RESTRICT l, Right const* RESTRICT r){
            BackendType:: template once_per_block<Id>::assign(l[Number],r[Number]);
            assign_strides_rec<Number-1, BackendType>::assign(l, r);
        }

    };

    /**@brief stopping the recursion*/
    template<typename BackendType>
    struct assign_strides_rec<0, BackendType>{
        static const uint_t Id=0;
        template<typename Left , typename Right >
        GT_FUNCTION
        static void assign(Left* RESTRICT l, Right const* RESTRICT r){
            BackendType:: template once_per_block<Id>::assign(l[0],r[0]);
        }
    };

    /**@brief recursively assigning the strides to a lobal array (i.e. m_strides).

       It implements the unrolling of a double loop: i.e. is n_f is the number of fields in this user function,
       and n_d(i) is the number of space dimensions per field (dependent on the ith field), then the loop for assigning the strides
       would look like
       for(i=0; i<n_f; ++i)
       for(j=0; j<n_d(i); ++j)
    */
    template<uint_t ID, typename BackendType>
    struct assign_strides{

        /**@brief does the actual assignment
         */
        template<typename Left, typename Right>
        GT_FUNCTION
        static void assign(Left& RESTRICT l, Right const& RESTRICT r){
#ifdef CXX11_ENABLED
            typedef typename std::remove_pointer< typename std::remove_reference<decltype(boost::fusion::at_c<ID>(r))>::type>::type storage_type;
#else
            typedef typename boost::remove_pointer< typename boost::remove_reference<BOOST_TYPEOF(boost::fusion::at_c<ID>(r))>::type>::type storage_type;
#endif
            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID < boost::mpl::size<Right>::value, "the ID is larger than the number of storage types")
                assign_strides_rec<storage_type::space_dimensions-2, BackendType>::assign(l.template get<ID>(), &boost::fusion::at_c<ID>(r)->strides(1));
            assign_strides<ID-1,BackendType>::assign(l,r); //tail recursion
        }
    };

    /**usual specialization to stop the recursion*/
    template<typename BackendType>
    struct assign_strides<0, BackendType>{
        template<typename Left, typename Right>
        GT_FUNCTION
        static void assign(Left & RESTRICT l_, Right const & RESTRICT r_){
#ifdef CXX11_ENABLED
            typedef typename std::remove_pointer< typename std::remove_reference<decltype(boost::fusion::at_c<0>(r_))>::type>::type storage_type;
#else
            typedef typename boost::remove_pointer< typename boost::remove_reference<BOOST_TYPEOF(boost::fusion::at_c<0>(r_))>::type>::type storage_type;
#endif
            assign_strides_rec<storage_type::space_dimensions-2, BackendType>::assign(l_.template get<0>(), &boost::fusion::at_c<0>(r_)->strides(1));
        }
    };

}//namespace gridtools
