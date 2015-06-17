#pragma once

#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include <boost/fusion/include/size.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/modulus.hpp>
#include "../gt_for_each/for_each.hpp"
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

    /**@brief alternative to boost::mlpl::range_c, which defines a sequence of integers of length End-Start, with step Step, in decreasing order*/
    template<typename Start, typename End, typename Step>
    struct gt_reversed_range{
        typedef typename boost::mpl::reverse_fold<
            boost::mpl::range_c<int_t, Start::value-1, End::value-1>//numeration from 0
            , boost::mpl::vector_c<int_t>
            , boost::mpl::push_back<boost::mpl::_1,
                                    boost::mpl::plus<boost::mpl::_2
                                                     , Step> > >::type type;
    };


    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template<typename T>
    struct is_any_iterate_domain_storage : boost::mpl::false_{};

    template < typename BaseStorage >
    struct is_any_iterate_domain_storage<storage<BaseStorage> > : boost::mpl::true_{};

    template <typename BaseStorage,
        uint_t TileI,
        uint_t TileJ,
        uint_t MinusI,
        uint_t MinusJ,
        uint_t PlusI,
        uint_t PlusJ
    >
    struct is_any_iterate_domain_storage<host_tmp_storage< BaseStorage, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ> > :
        boost::mpl::true_{};

    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template<typename T>
    struct is_any_iterate_domain_storage_pointer :
            boost::mpl::and_<
                is_any_iterate_domain_storage< typename boost::remove_pointer<T>::type >,
                boost::is_pointer<T>
            > {};

    /** method replacing the operator ? which selects a branch at compile time and
     allows to return different types whether the condition is true or false */
    template<bool Condition>
    struct static_if;

    template<>
    struct static_if <true>{
        template <typename TrueVal, typename FalseVal>
        static constexpr TrueVal& apply(TrueVal& true_val, FalseVal& /*false_val*/)
            {
                return true_val;
            }
    };

    template<>
    struct static_if <false>{
        template <typename TrueVal, typename FalseVal>
        static constexpr FalseVal& apply(TrueVal& /*true_val*/, FalseVal& false_val)
            {
                return false_val;
            }
    };

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
        typedef array<uint_t, storage_type::space_dimensions-1> data_array_t;

#ifdef CXX11_ENABLED
        template <short_t Idx>
            using return_t = typename boost::mpl::if_<boost::mpl::bool_<Idx==ID>, data_array_t, typename super::template return_t<Idx> >::type;
#else
        template <short_t Idx>
            struct return_t{
            typedef typename boost::mpl::if_<boost::mpl::bool_<Idx==ID>, data_array_t, typename super::template return_t<Idx>::type >::type type;
        };
#endif

        /**@brief constructor, doing nothing more than allocating the space*/
        GT_FUNCTION
        strides_cached():super(){
            GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<StorageList>::value > ID, "Library internal error: strides index exceeds the number of storages")
        }

        template<short_t Idx>
        GT_FUNCTION
#ifdef CXX11_ENABLED
            return_t<Idx>
#else
            typename return_t<Idx>::type
#endif
            & RESTRICT
            get() {
            return static_if<(Idx==ID)>::apply( m_data , super::template get<Idx>());
        }

    private:

        data_array_t m_data;
    };


    /**specialization to stop the recursion*/
    template<typename storage_list>
    struct strides_cached<0, storage_list>  {
        typedef typename boost::mpl::at_c<storage_list, 0>::type::storage_type storage_type;

        GT_FUNCTION
        strides_cached(){}

        typedef array<uint_t, storage_type::space_dimensions-1> data_array_t;

        template <short_t Idx>
#ifdef CXX11_ENABLED
        using return_t=data_array_t;
#else
        struct return_t{
            typedef data_array_t type;
        };
#endif
        //TODOCOSUNA getter should be const method. But we can not here because we return a non const *
        // We should have a getter and a setter
        template<short_t Idx>
        GT_FUNCTION
        data_array_t & RESTRICT
        get()  {//stop recursion
            return m_data;
        }

    private:
        data_array_t m_data;
    };

    template<typename T> struct is_strides_cached : boost::mpl::false_{};

    template<uint_t ID, typename StorageList>
    struct is_strides_cached< strides_cached<ID, StorageList> > : boost::mpl::true_{};

    //defines how many threads participate to the (shared) memory initialization
    //TODOCOSUNA This IS VERY VERY VERY DANGEROUS HERE
#define BLOCK_SIZE 32

    /**@brief functor assigning the 'raw' data pointers to an input data pointers array (i.e. the m_data_pointers array).

       The 'raw' datas are the one or more data fields contained in each of the storage classes used by the current user function.
       @tparam Offset an index identifying the starting position in the data pointers array of the portion corresponding to the given storage
       @tparam BackendType the type of backend
       @tparam StrategyType the type of strategy
       @tparam DataPointerArray gridtools array of data pointers
       @tparam Storage any of the storage type handled by the iterate domain
       To clarify the meaning of the two template indices, supposing that we have a 'rectangular' vector field, NxM, where N is the constant number of
       snapshots per storage, while M is the number of storages. Then 'Number' would be an index between 0 and N, while Offset would have the form n*M, where
       0<n<N is the index of the previous storage.
    */
    template<uint_t Offset, typename BackendType, typename DataPointerArray, typename Storage>
    struct assign_raw_data_functor{
        GRIDTOOLS_STATIC_ASSERT((is_array<DataPointerArray>::value),
                "Internal Error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_any_iterate_domain_storage<Storage>::value),
                "Internal Error: wrong type")

    private:
        DataPointerArray& RESTRICT m_data_pointer_array;
        Storage const * RESTRICT m_storage;
        const uint_t m_EU_id_i;
        const uint_t m_EU_id_j;

    public:

        GT_FUNCTION
        assign_raw_data_functor(DataPointerArray& RESTRICT data_pointer_array, Storage const * RESTRICT storage,
                const uint_t EU_id_i, const uint_t EU_id_j) :
                m_data_pointer_array(data_pointer_array), m_storage(storage), m_EU_id_i(EU_id_i), m_EU_id_j(EU_id_j) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            assert(m_storage);
            //compute the processing element in charge of doing the copy (i.e. the core in a backend with multiple cores)
            typedef typename boost::mpl::modulus<ID, boost::mpl::int_<BLOCK_SIZE> >::type pe_id_t;
            //provide the implementation that performs the assignment, depending on the type of storage we have
            impl<ID, pe_id_t, Storage>();
        }
    private:

        assign_raw_data_functor();

        // implementation of the assignment of the data pointer in case the storage is a temporary storage
        template<typename ID, typename PE_ID, typename _Storage>
        GT_FUNCTION
        void impl(typename boost::enable_if_c<is_host_tmp_storage<_Storage>::value>::type* = 0) const
        {
            BackendType::template once_per_block<PE_ID::value>::assign(
                    m_data_pointer_array[Offset+ID::value],m_storage->fields_offset(ID::value,m_EU_id_i, m_EU_id_j));
        }

        // implementation of the assignment of the data pointer in case the storage is a a regular storage
        template<typename ID, typename PE_ID, typename _Storage>
        GT_FUNCTION
        void impl(typename boost::disable_if_c<is_host_tmp_storage<_Storage>::value>::type* = 0) const
        {
            BackendType::template once_per_block<PE_ID::value>::assign(
                    m_data_pointer_array[Offset+ID::value], m_storage->fields()[ID::value].get());
        }
    };

    /**@brief metafunction that counts the total number of data fields which are neceassary for this functor (i.e. number of storage
     * instances times number of fields per storage)
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
                typename gt_reversed_range< static_int<0>, static_int<EndIndex>, static_int<1> >::type,
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

       @tparam Coordinate direction along which the increment takes place
       @tparam Execution policy determining how the increment is done (e.g. increment/decrement)
       @tparam StridesCached strides cached type
       @tparam StorageSequence sequence of storages

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
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
                "internal error: wrong type")

        GT_FUNCTION
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

    /**@brief functor initializing the indeces
     *     does the actual assignment
     *     This method is responsible of computing the index for the memory access at
     *     the location (i,j,k). Such index is shared among all the fields contained in the
     *     same storage class instance, and it is not shared among different storage instances.
     * @tparam Coordinate direction along which the increment takes place
     * @tparam StridesCached strides cached type
     * @tparam StorageSequence sequence of storages
     */
    template<uint_t Coordinate, typename Strides, typename StorageSequence>
    struct initialize_index_functor {
    private:
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
                "internal error: wrong type")


        Strides& RESTRICT m_strides;
        StorageSequence const & RESTRICT m_storages;
        const uint_t m_initial_pos;
        const uint_t m_block;
        uint_t* RESTRICT m_index_array;

    public:
        GT_FUNCTION
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

    /**@brief functor assigning all the storage pointers to the m_data_pointers array
     * This method is responsible of copying the base pointers of the storages inside a local vector
     * which is tipically instantiated on a fast local memory.
     *
     * The EU stands for ExecutionUnit (thich may be a thread or a group of
     * threads. There are potentially two ids, one over i and one over j, since
     * our execution model is parallel on (i,j). Defaulted to 1.
     * @tparam BackendType the type of backend
     * @tparam DataPointerArray gridtools array of data pointers
     * @tparam StorageSequence sequence of any of the storage types handled by the iterate domain
     * */
    template<typename BackendType, typename DataPointerArray, typename StorageSequence>
    struct assign_storage_functor{

        GRIDTOOLS_STATIC_ASSERT((is_array<DataPointerArray>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
                                "internal error: wrong type")
    private:
        DataPointerArray& RESTRICT m_data_pointer_array;
        StorageSequence const & RESTRICT m_storages;
        const int_t m_EU_id_i;
        const int_t m_EU_id_j;

    public:
        GT_FUNCTION
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

            for_each< typename gt_reversed_range< static_int<0>, static_int<storage_type::field_dimensions>, static_int<1> >::type > (
                assign_raw_data_functor<
                    total_storages<StorageSequence, ID::value>::value,
                    BackendType,
                    DataPointerArray,
                    storage_type
                >(m_data_pointer_array, boost::fusion::at<ID>(m_storages), m_EU_id_i, m_EU_id_j)
            );
        }
    };

    /**
       @brief functor assigning the storage strides to the m_strides array.
       This is the unrolling of the inner nested loop

       @tparam BackendType the type of backend
    */
    template<typename BackendType>
    struct assign_strides_inner_functor
    {
    private:
        uint_t* RESTRICT m_l;
        const uint_t* RESTRICT m_r;

    public:

        GT_FUNCTION
        assign_strides_inner_functor(uint_t* RESTRICT l, const uint_t* RESTRICT r) :
            m_l(l), m_r(r) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            assert(m_l);
            assert(m_r);
            const uint_t pe_id=(ID::value)%BLOCK_SIZE;

            BackendType:: template once_per_block<pe_id>::assign(m_l[ID::value],m_r[ID::value]);
        }
    };

    /**@brief functor assigning the strides to a lobal array (i.e. m_strides).

       It implements the unrolling of a double loop: i.e. is n_f is the number of fields in this user function,
       and n_d(i) is the number of space dimensions per field (dependent on the ith field), then the loop for assigning the strides
       would look like
       for(i=0; i<n_f; ++i)
       for(j=0; j<n_d(i); ++j)
     * @tparam BackendType the type of backend
     * @tparam StridesCached strides cached type
     * @tparam StorageSequence sequence of storages
    */
    template<typename BackendType, typename StridesCached, typename StorageSequence>
    struct assign_strides_functor{

        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "internal error: wrong type")
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
                "internal error: wrong type")

    private:
        StridesCached& RESTRICT m_strides;
        const StorageSequence& RESTRICT m_storages;

    public:
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

                for_each<typename gt_reversed_range< static_int<0>, static_int< storage_type::space_dimensions-1>, static_int<1> >::type> (
                assign_strides_inner_functor<
                BackendType
                >(&(m_strides.template get<ID::value>()[0]), &boost::fusion::at<ID>(m_storages)->strides(1))
                );
        }
    };

}//namespace gridtools
