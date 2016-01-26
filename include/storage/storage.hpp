#pragma once
#include "data_field.hpp"
#include "meta_storage.hpp"
#include "common/gpu_clone.hpp"
#include "common/generic_metafunctions/reverse_pack.hpp"

/**
@file
@brief Storage class
This extra layer is added on top of the base_storage class because it extends the clonabl_to_gpu interface. Due to the multiple inheritance pattern this class should not be further inherited.
*/

namespace gridtools{

    template < typename BaseStorage >
      struct storage : public BaseStorage, clonable_to_gpu<storage<BaseStorage> >
    {
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef storage<BaseStorage> original_storage;
        typedef clonable_to_gpu<storage<BaseStorage> > gpu_clone;
        typedef typename BaseStorage::iterator_type iterator_type;
        typedef typename BaseStorage::value_type value_type;
        static const ushort_t n_args = basic_type::n_width;
        static const ushort_t space_dimensions = basic_type::space_dimensions;
    private:
        typename super::meta_data_t const* m_device_storage_info;
    public:

        void clone_to_device() {
            //storage_info has to be cloned first
            assert(m_device_storage_info);
            m_device_storage_info = m_device_storage_info->device_pointer();
            clonable_to_gpu<storage<BaseStorage> >::clone_to_device();
        }

        /** @brief updates the CPU pointer */
        void d2h_update(){
            super::d2h_update();
            m_device_storage_info = (&this->meta_data());
        }

        __device__
        storage(storage const& other)
            :  super(other)
            , m_device_storage_info(other.m_device_storage_info)
        {}

        GT_FUNCTION
        typename super::meta_data_t* device_storage_info(){
            return m_device_storage_info;
        }

#if defined(CXX11_ENABLED)
        //forwarding constructor
        template <class ... ExtraArgs>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_
                           , ExtraArgs const& ... args )
            :super(meta_data_, args ...)
            , m_device_storage_info(&meta_data_)
        {
        }
#else

        template<typename T>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, T const& arg1 )
            :super(meta_data_, arg1)
            , m_device_storage_info(meta_data_.device_pointer())
            {
            }


        template <class T, class U>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, T const& arg1, U const& arg2 )
            :super(meta_data_, (value_type) arg1, arg2)
            , m_device_storage_info(meta_data_.device_pointer())

        {
        }

        template <class T, class U>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, T * arg1, U const& arg2 )
            :super(meta_data_, (value_type)* arg1, arg2)
            , m_device_storage_info(meta_data_.device_pointer())
        {
        }

#endif

//    private :
        explicit storage(typename basic_type::meta_data_t const& meta_data_)
            :super(meta_data_)
            , m_device_storage_info(meta_data_.device_pointer())
        {}

#ifdef CXX11_ENABLED


        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
         */
        template <typename ... UInt>
        GT_FUNCTION
        value_type& operator()(UInt const& ... dims) {
            //failure here means that you didn't call clone_to_device on the storage_info yet
            // assert(m_device_storage_info);
            return super::operator()(m_device_storage_info, dims...);
        }

        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
         */
        template <typename ... UInt>
        GT_FUNCTION
        value_type const & operator()(UInt const& ... dims) const {
            //failure here means that you didn't call clone_to_device on the storage_info yet
            //assert(m_device_storage_info);
            return super::operator()(m_device_storage_info, dims...);
        }
#else //CXX11_ENABLED

        /**
            @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
*/
        GT_FUNCTION
        value_type& operator()( uint_t const& i, uint_t const& j, uint_t const& k) {
            //failure here means that you didn't call clone_to_device on the storage_info yet
            //assert(m_device_storage_info);
            return super::operator()(m_device_storage_info, i,j,k);
        }


        /**
            @brief returns (by const reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
        */
        GT_FUNCTION
        value_type const & operator()( uint_t const& i, uint_t const& j, uint_t const& k) const {
            //failure here means that you didn't call clone_to_device on the storage_info yet
            //assert(m_device_storage_info);
            return super::operator()(m_device_storage_info, i,j,k);
        }
#endif



    /**@brief swaps two arbitrary snapshots in two arbitrary data field dimensions

       @tparam SnapshotFrom one snapshot
       @tparam DimFrom one dimension
       @tparam SnapshotTo the second snapshot
       @tparam DimTo the second dimension

       syntax:
       swap<3,1>::with<4,1>::apply(storage_);
    */
    template<ushort_t SnapshotFrom, ushort_t DimFrom=0>
    struct swap{
        template<ushort_t SnapshotTo, ushort_t DimTo=0>
        struct with{

            template<typename Storage>
            GT_FUNCTION
            static void apply(Storage& storage_){
                super::template swap<SnapshotFrom, DimFrom>::template with<SnapshotTo, DimTo>::apply(storage_);
                storage_.clone_to_device();
            }
        };
    };

    };

    /**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each dimension has arbitrary size 'Number'.

       Annoyngly enough does not work with CUDA 6.5
    */
#if defined(CXX11_ENABLED)

    /** @brief syntactic sugar for defining a data field

        Given a storage type and the dimension number it generates the correct data field type
        @tparam Storage the basic storage used
        @tparam Number the number of snapshots in each dimension
     */
    template< class Storage, uint_t ... Number >
    struct field_reversed;

    /**
     @brief specialization for the GPU storage
     the defined type is storage (which is clonable_to_gpu)
    */
    template< class BaseStorage, uint_t ... Number >
    struct field_reversed<storage<BaseStorage>, Number ... >{
        typedef storage< data_field< storage_list<base_storage<typename BaseStorage::pointer_type, typename  BaseStorage::meta_data_t, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    };

    /**
       @brief specialization for the CPU storage (base_storage)
       the difference being that the type is directly the base_storage (which is not clonable_to_gpu)
    */
    template< class PointerType, class MetaData, ushort_t FD, uint_t ... Number >
    struct field_reversed<base_storage<PointerType, MetaData, FD>, Number ... >{
        typedef data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, GPU storage)*/
    template<  typename PointerType
               ,typename MetaData
               ,ushort_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<storage<base_storage<PointerType, MetaData, FieldDimension> > >, Number... >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, CPU storage)*/
    template<  typename PointerType
               ,typename MetaData
               ,ushort_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<base_storage<PointerType, MetaData, FieldDimension> >, Number... >{
        typedef no_storage_type_yet<data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > type;
    };

    /**@brief interface for definig a data field

       @tparam Storage the basic storage type shared by all the snapshots
       @tparam First  all the subsequent parameters define the dimensionality of the snapshot arrays
        in all the data field dimensions
     */
    template< class Storage, uint_t First, uint_t ... Number >
    struct field{
        typedef typename reverse_pack<Number ...>::template apply<field_reversed, Storage, First >::type::type type;
    };
#endif

    template <typename T>
    std::ostream& operator<<(std::ostream &s, storage<T> const & x ) {
        s << "storage< "
          << static_cast<T const&>(x) << " > ";
        return s;
    }

#ifdef CXX11_ENABLED
    template<typename T>
    struct is_storage : boost::mpl::or_<
        is_data_field<T>
        , is_storage_list<T> >{};
#else
    template<typename T>
    struct is_storage : boost::mpl::false_{};
#endif

    template<typename T>
    struct is_storage<storage<T> > : boost::mpl::true_{};

    template < typename PointerType, typename MetaData, ushort_t FieldDimension >
    struct is_storage<base_storage<PointerType, MetaData, FieldDimension> > : boost::mpl::true_{};

}//namespace gridtools
