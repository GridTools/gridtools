#pragma once

#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include <boost/fusion/include/size.hpp>
#include "expressions.h"
#include "accessor_metafunctions.h"
#include "../common/meta_array.h"
#include "../common/array.h"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools{

    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template<typename T>
    struct is_any_iterate_domain_storage : boost::mpl::false_{};

    template < typename BaseStorage >
    struct is_any_iterate_domain_storage<storage<BaseStorage>* > : boost::mpl::true_{};

    template <typename BaseStorage,
        uint_t TileI,
        uint_t TileJ,
        uint_t MinusI,
        uint_t MinusJ,
        uint_t PlusI,
        uint_t PlusJ
    >
    struct is_any_iterate_domain_storage<host_tmp_storage< BaseStorage, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>* > :
        boost::mpl::true_{};

    /**
       @brief struct to allocate recursively all the strides with the proper dimension

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageList typelist of the storages
    */
    //TODOCOSUNA this is just an array, no need for special class, looks like
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

        //TODOCOSUNA getter should be const method. But we can not here because we return a non const *
        // We should have a getter and a setter
        template<short_t Idx>
        GT_FUNCTION
        uint_t * RESTRICT get()  {//stop recursion
            //GRIDTOOLS_STATIC_ASSERT(Idx==0, "Internal library error: Index exceeding the storage_list dimension.")
            return &m_data[0];
        }

    private:
        uint_t m_data[storage_type::space_dimensions-1];
    };

    template<typename T> struct is_strides_cached : boost::mpl::false_{};

    template<uint_t ID, typename StorageList>
    struct is_strides_cached< strides_cached<ID, StorageList> > : boost::mpl::true_{};

    //defines how many threads participate to the (shared) memory initialization
    //TODOCOSUNA This IS VERY VERY VERY DANGEROUS HERE
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
                BackendType::template once_per_block<Id>::assign(l[Offset+Number],r->fields_offset(Number,EU_id_i, EU_id_j));
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
    template <typename StoragesVector, int_t EndIndex>
    struct total_storages{
        //the index must not exceed the number of storages
        GRIDTOOLS_STATIC_ASSERT(EndIndex <= boost::mpl::size<StoragesVector>::type::value,
                "the index must not exceed the number of storages")

        template<typename Index_>
        struct get_field_dimensions{
            typedef typename boost::mpl::int_<
                 boost::remove_pointer<
                     typename boost::remove_reference<
                         typename boost::mpl::at<StoragesVector, Index_ >::type
                     >::type
                 >::type::field_dimensions
             >::type type;
        };

        typedef typename boost::mpl::if_c<
            (EndIndex < 0),
            boost::mpl::int_<0>,
            typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, EndIndex>,
                boost::mpl::int_<0>,
                boost::mpl::plus<
                    boost::mpl::_1,
                    get_field_dimensions<boost::mpl::_2>
                >
            >::type
        >::type type;

        static const uint_t value=type::value;
    };

    /**@brief incrementing all the storage pointers to the m_data_pointers array

       \tparam ID identifier for the current storage (recursion index)
       \tparam Coordinate direction along which the increment takes place
       \tparam Execution policy determining how the increment is done (e.g. increment/decrement)

           This method is responsible of incrementing the index for the memory access at
           the location (i,j,k) incremented/decremented by 1 along the 'Coordinate' direction. Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           The actual increment computation is delegated to the storage classes, the reason being that the implementation may depend on the storage type
           (e.g. whether the storage is temporary, partiitoned into blocks, ...)
    */
    template<
        uint_t Coordinate,
        enumtype::execution Execution,
        typename StridesCached,
        typename StorageSequence>
    struct increment_index_functor {

        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage>::value),
                "internal error: wrong type")

        increment_index_functor(StorageSequence const& storages, uint_t const& increment,
                uint_t* RESTRICT index_array, StridesCached &  RESTRICT strides_cached) :
            m_storages(storages), m_increment(increment), m_index_array(index_array), m_strides_cached(strides_cached){}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<StorageSequence>::value),
                    "Accessing an index out of bound in fusion tuple")

            assert(m_index_array);
            boost::fusion::at<ID>(m_storages)->template increment<Coordinate, Execution>(
                    m_increment,&m_index_array[ID::value], m_strides_cached.template get<ID::value>());
        }
    private:
        StorageSequence const& m_storages;
        uint_t const& m_increment;
        uint_t* RESTRICT m_index_array;
        StridesCached &  RESTRICT m_strides_cached;
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
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array")
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
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array")
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
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array")
            index[0]=id;
        }

        template<typename Array>
        GT_FUNCTION
        static void set(Array const& index, Array& out){
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array")
            out[0]=index[0];
        }
    };

    /**@brief initializing the indeces
     *     does the actual assignment
           This method is responsible of computing the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.
     *
     */
    template<uint_t Coordinate, typename Strides, typename StorageSequence>
    struct initialize_index_functor {
    private:
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage>::value),
                "internal error: wrong type")


        Strides& RESTRICT m_strides;
        StorageSequence const & RESTRICT m_storages;
        const uint_t m_initial_pos;
        const uint_t m_block;
        uint_t* RESTRICT m_index_array;

    public:
        initialize_index_functor(Strides& RESTRICT strides, StorageSequence const & RESTRICT storages, const uint_t initial_pos,
            const uint_t block, uint_t* RESTRICT index_array) :
            m_strides(strides), m_storages(storages), m_initial_pos(initial_pos), m_block(block),
            m_index_array(index_array) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<StorageSequence>::value),
                "Accessing an index out of bound in fusion tuple")

            assert(m_index_array);

            boost::fusion::at<ID>(m_storages)->template initialize<Coordinate>(
                    m_initial_pos, m_block, m_strides.template get<ID::value>(), &m_index_array[ID::value]);
        }
    };

    /**@brief assigning all the storage pointers to the m_data_pointers array
     * This method is responsible of copying the base pointers of the storages inside a local vector
     * which is tipically instantiated on a fast local memory.
     *
     * The EU stands for ExecutionUnit (thich may be a thread or a group of
     * threads. There are potentially two ids, one over i and one over j, since
     * our execution model is parallel on (i,j). Defaulted to 1.
     * */
    template<typename BackendType, typename DataPointerArray, typename StorageSequence>
    struct assign_storage_functor{

        GRIDTOOLS_STATIC_ASSERT((is_array<DataPointerArray>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage>::value),
                "internal error: wrong type")

        assign_storage_functor(DataPointerArray& RESTRICT data_pointer_array, StorageSequence const& RESTRICT storages,
                const int_t EU_id_i, const int_t EU_id_j) :
            m_data_pointer_array(data_pointer_array), m_storages(storages), m_EU_id_i(EU_id_i), m_EU_id_j(EU_id_j) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<StorageSequence>::value),
                    "Accessing an index out of bound in fusion tuple")

            typedef typename boost::remove_pointer<
                typename boost::remove_reference<
                    typename boost::fusion::result_of::at<StorageSequence, ID>::type
                 >::type
            >::type storage_type;

            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size<StorageSequence>::value,
                    "the ID is larger than the number of storage types")

            assign_raw_data<storage_type::field_dimensions-1,
                total_storages<StorageSequence, ID::value>::value,
                BackendType,
                storage_type
            >::assign(m_data_pointer_array, boost::fusion::at<ID>(m_storages), m_EU_id_i, m_EU_id_j);

        }
    private:
        DataPointerArray& RESTRICT m_data_pointer_array;
        StorageSequence const & RESTRICT m_storages;
        const int_t m_EU_id_i;
        const int_t m_EU_id_j;

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
    template<typename BackendType, typename StridesCached, typename StorageSequence>
    struct assign_strides_functor{

        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage>::value),
                "internal error: wrong type")

        assign_strides_functor(StridesCached& RESTRICT strides, StorageSequence const& RESTRICT storages) :
            m_strides(strides), m_storages(storages) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<StorageSequence>::value),
                    "Accessing an index out of bound in fusion tuple")

            typedef typename boost::remove_pointer<
                typename boost::remove_reference<
                    typename boost::fusion::result_of::at<StorageSequence, ID>::type
                 >::type
            >::type storage_type;

            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size<StorageSequence>::value, "the ID is larger than the number of storage types")

            assign_strides_rec<storage_type::space_dimensions-2, BackendType>::assign(
                    m_strides.template get<ID::value>(), &boost::fusion::at<ID>(m_storages)->strides(1));
        }
    private:
        StridesCached& RESTRICT m_strides;
        const StorageSequence& RESTRICT m_storages;
    };

}//namespace gridtools
