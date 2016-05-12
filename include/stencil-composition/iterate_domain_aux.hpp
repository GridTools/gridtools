#pragma once

#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include <boost/fusion/include/size.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/modulus.hpp>
#include "../gt_for_each/for_each.hpp"
#include "expressions.hpp"
#include "accessor_metafunctions.hpp"
#include "../common/meta_array.hpp"
#include "../common/array.hpp"
#include "common/generic_metafunctions/static_if.hpp"
#include "common/generic_metafunctions/reversed_range.hpp"

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
    struct is_any_iterate_domain_storage<storage<BaseStorage> > : boost::mpl::true_{};

    template <typename T> struct meta_storage_derived;

    template<typename T>
    struct is_any_iterate_domain_meta_storage : boost::mpl::false_{};

    template < typename BaseStorage >
    struct is_any_iterate_domain_meta_storage<meta_storage_derived<BaseStorage> > : boost::mpl::true_{};

    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template<typename T>
    struct is_any_iterate_domain_storage_pointer :
            boost::mpl::and_<
                is_any_iterate_domain_storage< typename boost::remove_pointer<T>::type >,
                boost::is_pointer<T>
            > {};

    template<typename T>
    struct is_any_iterate_domain_meta_storage_pointer :
            boost::mpl::and_<
                is_any_iterate_domain_meta_storage< typename boost::remove_pointer<T>::type >,
                boost::is_pointer<T>
            > {};

    /**
       @brief struct to allocate recursively all the strides with the proper dimension

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageList typelist of the storages
    */
    //TODOCOSUNA this is just an array, no need for special class, looks like
    template<ushort_t ID, typename StorageList>
    struct strides_cached : public strides_cached<ID-1, StorageList> {
        typedef typename  boost::mpl::at_c<StorageList, ID>::type storage_type;
        typedef strides_cached<ID-1, StorageList> super;
        typedef array<int_t, storage_type::space_dimensions-1> data_array_t;

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
            GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<StorageList>::value > ID, "Library internal error: strides index exceeds the number of storages");
        }

        template<short_t Idx>
        GT_FUNCTION
#ifdef CXX11_ENABLED
        return_t<Idx>
#else
        typename return_t<Idx>::type
#endif
        const & RESTRICT
        get() const {
            return static_if<(Idx==ID)>::apply(
                    m_data ,
                    super::template get<Idx>()
            );
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
        strides_cached(strides_cached const&);
    };


    /**specialization to stop the recursion*/
    template<typename MetaStorageList>
        struct strides_cached<(ushort_t)0, MetaStorageList>  {
        typedef typename boost::mpl::at_c<MetaStorageList, 0>::type storage_type;

        GT_FUNCTION
        strides_cached(){}

        typedef array<int_t, storage_type::space_dimensions-1> data_array_t;

        template <short_t Idx>
#ifdef CXX11_ENABLED
        using return_t=data_array_t;
#else
        struct return_t{
            typedef data_array_t type;
        };
#endif

        template<short_t Idx>
        GT_FUNCTION
        data_array_t & RESTRICT
        get()  {//stop recursion
            return m_data;
        }

        template<short_t Idx>
        GT_FUNCTION
        data_array_t const & RESTRICT
        get() const {//stop recursion
            return m_data;
        }

    private:
        data_array_t m_data;
        strides_cached(strides_cached const&);
    };

    template<typename T> struct is_strides_cached : boost::mpl::false_{};

    template<uint_t ID, typename StorageList>
    struct is_strides_cached< strides_cached<ID, StorageList> > : boost::mpl::true_{};

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
                                "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_any_iterate_domain_storage<Storage>::value),
                                "Internal Error: wrong type");

    private:
        DataPointerArray& RESTRICT m_data_pointer_array;
        Storage const * RESTRICT m_storage;
        const uint_t m_offset;

    public:
        GT_FUNCTION
        assign_raw_data_functor( assign_raw_data_functor const& other): m_data_pointer_array(other.m_data_pointer_array), m_storage(other.m_storage), m_offset(other.m_offset){}

        GT_FUNCTION
        assign_raw_data_functor(DataPointerArray& RESTRICT data_pointer_array, Storage const * RESTRICT storage, uint const offset_) :
            m_data_pointer_array(data_pointer_array), m_storage(storage), m_offset(offset_) {}

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
        void impl() const
        {
            BackendType::template once_per_block<PE_ID::value>::assign(
                m_data_pointer_array[Offset+ID::value], m_storage->template access_value<ID>()+m_offset);
        }
    };

    /**@brief metafunction that counts the total number of data fields which are neceassary for this functor (i.e. number of storage
     * instances times number of fields per storage)
    */
    template <typename StoragesVector, int_t EndIndex>
    struct total_storages{
        DISALLOW_COPY_AND_ASSIGN(total_storages);
        //the index must not exceed the number of storages
        GRIDTOOLS_STATIC_ASSERT(EndIndex <= boost::mpl::size<StoragesVector>::type::value,
                                "the index must not exceed the number of storages");

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
                typename reversed_range< int_t, 0, EndIndex >::type,
                boost::mpl::int_<0>,
                boost::mpl::plus<
                    boost::mpl::_1,
                    get_field_dimensions<boost::mpl::_2>
                    >
            >::type
        >::type type;

        static const uint_t value=type::value;
    private:
        total_storages();
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
        typename StridesCached,
        typename MetaStorageSequence
        >
    struct increment_index_functor {

        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "internal error: wrong type");
        // GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
        //                         "internal error: wrong type");

        GT_FUNCTION
        increment_index_functor(MetaStorageSequence const& storages, int_t const& increment,
                int_t* RESTRICT index_array, StridesCached &  RESTRICT strides_cached) :
            m_storages(storages), m_increment(increment), m_index_array(index_array), m_strides_cached(strides_cached){}

        template <typename Pair>
        GT_FUNCTION
        void operator()(Pair const&) const {

            typedef typename boost::mpl::second<Pair>::type ID;
            typedef typename boost::mpl::first<Pair>::type metadata_t;

            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<MetaStorageSequence>::value),
                                    "Accessing an index out of bound in fusion tuple");

            assert(m_index_array);
            boost::fusion::at_c<ID::value>(m_storages)->template increment<Coordinate>(
                m_increment,&m_index_array[ID::value], m_strides_cached.template get<ID::value>());
        }

        GT_FUNCTION
        increment_index_functor(increment_index_functor const& other) : m_storages(other.m_storages), m_increment(other.m_increment), m_index_array(other.m_index_array), m_strides_cached(other.m_strides_cached){};
    private:
        increment_index_functor();

        MetaStorageSequence const& m_storages;
        int_t const& m_increment;
        int_t* RESTRICT m_index_array;
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
        static void set(int_t const& id, Array& index){
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array");
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
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array");
            out[ID]=index[ID];
            set_index_recur<ID-1>::set(index, out);
        }
    private:
        set_index_recur();
        set_index_recur(set_index_recur const&);
    };


    /**usual specialization to stop the recursion*/
    template<>
    struct set_index_recur<0>{

        template<typename Array>
        GT_FUNCTION
        static void set( int_t const& id, Array& index/* , ushort_t* lru */){
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array");
            index[0]=id;
        }

        template<typename Array>
        GT_FUNCTION
        static void set(Array const& index, Array& out){
            GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "type is not a gridtools array");
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
    template<uint_t Coordinate, typename Strides, typename MetaStorageSequence>
    struct initialize_index_functor {
    private:
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), "internal error: wrong type");
        // GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
        //                         "internal error: wrong type");


        Strides& RESTRICT m_strides;
        MetaStorageSequence const & RESTRICT m_storages;
        const int_t m_initial_pos;
        const uint_t m_block;
        int_t* RESTRICT m_index_array;
        initialize_index_functor();
    public:
        GT_FUNCTION
        initialize_index_functor(initialize_index_functor const& other) : m_strides(other.m_strides), m_storages(other.m_storages), m_initial_pos(other.m_initial_pos), m_block(other.m_block), m_index_array(other.m_index_array){}

        GT_FUNCTION
        initialize_index_functor(Strides& RESTRICT strides, MetaStorageSequence const & RESTRICT storages, const int_t initial_pos,
            const uint_t block, int_t* RESTRICT index_array) :
            m_strides(strides), m_storages(storages), m_initial_pos(initial_pos), m_block(block),
            m_index_array(index_array) {}

        template <typename Pair>
        GT_FUNCTION
        void operator()(Pair const&) const {

            typedef typename boost::mpl::second<Pair>::type id_t;
            GRIDTOOLS_STATIC_ASSERT((id_t::value < boost::fusion::result_of::size<MetaStorageSequence>::value),
                                    "Accessing an index out of bound in fusion tuple");

            assert(m_index_array);

            boost::fusion::at<id_t>(m_storages)->template initialize<Coordinate>(
                m_initial_pos, m_block, &m_index_array[id_t::value], m_strides.template get<id_t::value>());
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
    template<typename BackendType, typename DataPointerArray, typename StorageSequence, typename MetaStorageSequence, typename MetaDataMap>
    struct assign_storage_functor{

        GRIDTOOLS_STATIC_ASSERT((is_array<DataPointerArray>::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<StorageSequence, is_any_iterate_domain_storage_pointer>::value),
                                "internal error: wrong type");
    private:
        DataPointerArray& RESTRICT m_data_pointer_array;
        StorageSequence const & RESTRICT m_storages;
        MetaStorageSequence const & RESTRICT m_meta_storages;
        const int_t m_EU_id_i;
        const int_t m_EU_id_j;
        assign_storage_functor();
    public:
        GT_FUNCTION
        assign_storage_functor(assign_storage_functor const& other): m_data_pointer_array(other.m_data_pointer_array), m_storages(other.m_storages), m_meta_storages(other.m_meta_storages), m_EU_id_i(other.m_EU_id_i), m_EU_id_j(other.m_EU_id_j)  {}

        GT_FUNCTION
        assign_storage_functor(DataPointerArray& RESTRICT data_pointer_array, StorageSequence const& RESTRICT storages, MetaStorageSequence const& RESTRICT meta_storages,
                const int_t EU_id_i, const int_t EU_id_j) :
            m_data_pointer_array(data_pointer_array), m_storages(storages), m_meta_storages(meta_storages), m_EU_id_i(EU_id_i), m_EU_id_j(EU_id_j) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size<StorageSequence>::value),
                                    "Accessing an index out of bound in fusion tuple");

            typedef typename boost::remove_pointer<
                typename boost::remove_reference<
                    typename boost::fusion::result_of::at<StorageSequence, ID>::type
                 >::type
            >::type storage_type;

            typedef typename boost::mpl::at
                <MetaDataMap, typename storage_type::meta_data_t >::type metadata_index_t;

            pointer<const typename storage_type::meta_data_t> const metadata_ = boost::fusion::at
                < metadata_index_t >(m_meta_storages);



            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size<StorageSequence>::value,
                                    "the ID is larger than the number of storage types");

                for_each< typename reversed_range<short_t, 0, storage_type::field_dimensions >::type > (
                assign_raw_data_functor<
                    total_storages<StorageSequence, ID::value>::value,
                    BackendType,
                    DataPointerArray,
                    storage_type
                >(m_data_pointer_array, boost::fusion::at<ID>(m_storages), metadata_->fields_offset(m_EU_id_i, m_EU_id_j))
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
        int_t* RESTRICT m_left;
        const int_t* RESTRICT m_right;

    public:

        GT_FUNCTION
        assign_strides_inner_functor(int_t* RESTRICT l, const int_t* RESTRICT r) :
            m_left(l), m_right(r) {}

        template <typename ID>
        GT_FUNCTION
        void operator()(ID const&) const {
            assert(m_left);
            assert(m_right);
            const uint_t pe_id=(ID::value)%BLOCK_SIZE;
            BackendType:: template once_per_block<pe_id>::assign(m_left[ID::value],m_right[ID::value]);
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
     * @tparam MetaStorageSequence sequence of storages
    */
    template<typename BackendType, typename StridesCached, typename MetaStorageSequence>
    struct assign_strides_functor{

        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "internal error: wrong type");
        // GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MetaStorageSequence, is_any_iterate_domain_meta_storage_pointer>::value),
        //                         "internal error: wrong type");

    private:
        StridesCached& RESTRICT m_strides;
        const MetaStorageSequence& RESTRICT m_storages;
        assign_strides_functor();

    public:
        GT_FUNCTION
        assign_strides_functor(assign_strides_functor const& other): m_strides(other.m_strides), m_storages(other.m_storages) {}

        GT_FUNCTION
        assign_strides_functor(StridesCached& RESTRICT strides, MetaStorageSequence const& RESTRICT storages) :
            m_strides(strides), m_storages(storages) {}

        template <typename Pair>
        GT_FUNCTION
        void operator()(Pair const&) const {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::second<Pair>::type::value < boost::fusion::result_of::size<MetaStorageSequence>::value),
                                    "Accessing an index out of bound in fusion tuple");

            typedef typename boost::mpl::second<Pair>::type ID;

            typedef typename boost::mpl::first<Pair>::type meta_storage_type;

            //if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size<MetaStorageSequence>::value, "the ID is larger than the number of storage types");

#ifdef CXX11_ENABLED
#ifndef __CUDACC__
            GRIDTOOLS_STATIC_ASSERT((std::remove_reference<decltype(m_strides.template get<ID::value>())>::type::size()==meta_storage_type::space_dimensions-1), "internal error: the length of the strides vectors does not match. The bug fairy has no mercy.");
#endif
#endif
            for_each< boost::mpl::range_c< short_t, 0,  meta_storage_type::space_dimensions-1> > (
            assign_strides_inner_functor<BackendType>
            (&(m_strides.template get<ID::value>()[0]), &(boost::fusion::template at_c<ID::value>(m_storages)->strides(1)))
                );
        }
    };

    template<typename Accessor, typename CachesMap>
    struct accessor_is_cached
    {
        typedef typename boost::mpl::eval_if<
            is_accessor<Accessor>,
            accessor_index<Accessor>,
            boost::mpl::identity<static_int<-1> >
        >::type accessor_index_t;

        typedef typename boost::mpl::eval_if<
            is_accessor<Accessor>,
            boost::mpl::has_key<
                CachesMap,
                //TODO: ERROR in Clang:
                //non-type template argument evaluates to -1, which cannot be narrowed to type 'uint_t'
#ifdef __CUDACC__
                static_uint<accessor_index_t::value>
#else // the following is NOT correct!! but compiles
                static_int<accessor_index_t::value>
#endif
                >,
            boost::mpl::identity<boost::mpl::false_>
        >::type type;
        BOOST_STATIC_CONSTANT(bool, value=(type::value));
    };


}//namespace gridtools
