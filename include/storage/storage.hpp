#pragma once
#include "data_field.hpp"
#include "common/gpu_clone.hpp"
#ifdef CXX11_ENABLED
#include "common/generic_metafunctions/reverse_pack.hpp"
#endif

/**
@file
@brief Storage class
This extra layer is added on top of the base_storage class because it extends the clonabl_to_gpu interface. Due to the
multiple inheritance pattern this class should not be further inherited.
*/

namespace gridtools {

    template < typename BaseStorage >
    struct storage : public BaseStorage, clonable_to_gpu< storage< BaseStorage > > {
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename basic_type::storage_info_type storage_info_type;
        typedef storage< BaseStorage > original_storage;
        typedef clonable_to_gpu< storage< BaseStorage > > gpu_clone;
        typedef typename BaseStorage::iterator_type iterator_type;
        typedef typename BaseStorage::value_type value_type;
        static const ushort_t n_args = basic_type::n_width;
        static const ushort_t space_dimensions = basic_type::space_dimensions;

      private:
        typename super::storage_info_type const *m_device_storage_info;
        bool m_on_host;

        /**@brief change the storage state to host

           do nothing if the storage is already in host stage
         */
        GT_FUNCTION
        void on_host() {
            if (!m_on_host) {
                m_device_storage_info = (&this->meta_data());
                m_on_host = true;
            }
        }

        /**@brief change the storage state to device

           do nothing if the storage is already in device stage
         */
        GT_FUNCTION
        void on_device() {
            if (m_on_host) {
                m_device_storage_info = this->m_meta_data.device_pointer();
                m_on_host = false;
            }
        }

      public:
        /**@brief calls the device copy constructor to update the device copy of the storage object.

           Such copy constructor is not supposed to copy the storage data. A shallow copy is made of the pointers
           contained in the m_fields data member. For this reason the call to clone_to_gpu has as side-effect a small
           kernel launch, but
           it is not a costly transfer from the host to the device.
        */
        void clone_to_device() {

            // assert(m_device_storage_info);
            // assert(m_device_storage_info->device_pointer());
            // assert(this->m_fields[0].get());
            on_device();
            clonable_to_gpu< storage< BaseStorage > >::clone_to_device();
        }

        /** @brief updates the CPU pointer */
        __host__ void d2h_update() {
            super::d2h_update();
            on_host();
        }

        /** @brief updates the CPU pointer */
        __host__ void h2d_update() {
            super::h2d_update();
            on_device();
        }

        /**@brief device copy constructor

           used by clone_to_device, to copy the object to the device (see \ref gridtools::clonable_to_gpu)
           NOTE: the actual raw data of the storage is not copied, a shallow copy is made for the pointers in the
           m_fields data member.
         */
        __device__ storage(storage const &other)
            : super(other), m_device_storage_info(other.m_device_storage_info), m_on_host(false) {}

        GT_FUNCTION
        typename super::storage_info_type const *device_storage_info() const { return m_device_storage_info; }

#if defined(CXX11_ENABLED)
        // forwarding constructor
        template < class... ExtraArgs >
        explicit storage(typename basic_type::storage_info_type const &meta_data_, ExtraArgs const &... args)
            : super(meta_data_, args...), m_device_storage_info(&meta_data_), m_on_host(true) {}
#else // CXX11_ENABLED

        template < typename T >
        explicit storage(typename basic_type::storage_info_type const &meta_data_, T const &arg1)
            : super(meta_data_, arg1), m_device_storage_info(&meta_data_), m_on_host(true) {}

        template < class T, class U >
        explicit storage(typename basic_type::storage_info_type const &meta_data_, T const &arg1, U const &arg2)
            : super(meta_data_, (value_type)arg1, arg2), m_device_storage_info(&meta_data_), m_on_host(true) {}

        template < class T, class U >
        explicit storage(typename basic_type::storage_info_type const &meta_data_, T *arg1, U const &arg2)
            : super(meta_data_, (value_type)*arg1, arg2), m_device_storage_info(&meta_data_), m_on_host(true) {}

#endif // CXX11_ENABLED

        //    private :
        explicit storage(typename basic_type::storage_info_type const &meta_data_)
            : super(meta_data_), m_device_storage_info(&meta_data_) {}

#ifdef CXX11_ENABLED

        /**
           explicitly disables the case in which the storage_info is passed by copy.
        */
        template < typename... T >
        storage(typename basic_type::storage_info_type &&, T...) = delete;

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
         */
        template < typename... UInt >
        GT_FUNCTION value_type &operator()(UInt const &... dims) {
// failure here means that you didn't call clone_to_device on the storage_info yet
#ifdef __CUDA_ARCH__
            assert(!m_on_host);
#else  //__CUDA_ARCH__
#ifndef NDEBUG
            if (!m_on_host)
                exit(-1);
            if (!m_device_storage_info)
                exit(-2);
#endif
// assert(m_on_host);
// assert(m_device_storage_info);
#endif //__CUDA_ARCH__

            return access_data_impl(m_device_storage_info, dims...);
        }

        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
         */
        template < typename... UInt >
        GT_FUNCTION value_type const &operator()(UInt const &... dims) const {
// failure here means that you didn't call clone_to_device on the storage_info yet
#ifdef __CUDA_ARCH__
            assert(!m_on_host);
#else  //__CUDA_ARCH__
#ifndef NDEBUG
            if (!m_on_host)
                exit(-1);
            if (!m_device_storage_info)
                exit(-2);
#endif
// assert(m_on_host);
// assert(m_device_storage_info);
#endif //__CUDA_ARCH__

            return access_data_impl(m_device_storage_info, dims...);
        }

      private:
        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            This interface is not exposed to the user, it gets called from storage.hpp
         */
        template < typename... UInt >
        GT_FUNCTION value_type &access_data_impl(storage_info_type const *metadata_, UInt const &... dims) {
#ifdef __CUDA_ARCH__
// assert(metadata_ && metadata_->index(dims...) < metadata_->size());
// assert(this->is_set);
#else
#ifndef NDEBUG
            if (!metadata_ || !(metadata_->index(dims...) < metadata_->size()))
            {
                printf("%d < %d\n", metadata_->index(dims...), metadata_->size());
                exit(-1);
            }
            if (!this->is_set)
                exit(-2);
#endif
#endif
            return (this->m_fields[0])[metadata_->index(dims...)];
        }

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            This interface is not exposed to the user, it gets called from storage.hpp
         */
        template < typename... UInt >
        GT_FUNCTION value_type const &access_data_impl(storage_info_type const *metadata_, UInt const &... dims) const {
#ifdef __CUDA_ARCH__
            assert(metadata_ && metadata_->index(dims...) < metadata_->size());
            assert(this->is_set);
#else
#ifndef NDEBUG
            if (!metadata_ || !(metadata_->index(dims...) < metadata_->size()))
                exit(-1);
            if (!this->is_set)
                exit(-2);
#endif
#endif
            return (this->m_fields[0])[metadata_->index(dims...)];
        }

#else // CXX11_ENABLED

        /**
            @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
*/
        GT_FUNCTION
        value_type &operator()(uint_t const &i, uint_t const &j, uint_t const &k) {
#ifdef __CUDA_ARCH__
            assert(!m_on_host);
#else  //__CUDA_ARCH__
            // assert(m_on_host);
#ifndef NDEBUG
            if (!m_on_host)
                exit(-1);
            if (!m_device_storage_info)
                exit(-2);
#endif
#endif //__CUDA_ARCH__

            return access_data_impl(m_device_storage_info, i, j, k);
        }

        /**
            @brief returns (by const reference) the value of the data field at the coordinates (i, j, k)

            this api is callable from the device if the associated storage_info has been previously cloned to the device
        */
        GT_FUNCTION
        value_type const &operator()(uint_t const &i, uint_t const &j, uint_t const &k) const {

// failure here means that you didn't call clone_to_device on the storage_info yet
#ifdef __CUDA_ARCH__
            assert(!m_on_host);
#else  // __CUDA_ARCH__
            // assert(m_on_host);
#ifndef NDEBUG
            if (!m_on_host)
                exit(-1);
            if (!m_device_storage_info)
                exit(-2);
#endif
#endif //__CUDA_ARCH__

            return access_data_impl(m_device_storage_info, i, j, k);
        }

      private:
        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            This interface is not exposed to the user, it gets called from storage.hpp
         */
        GT_FUNCTION
        value_type &access_data_impl(
            storage_info_type const *metadata_, uint_t const &i, uint_t const &j, uint_t const &k) {
#ifdef __CUDA_ARCH__
            assert(metadata_ && metadata_->index(i, j, k) < metadata_->size());
            assert(this->is_set);
#else
#ifndef NDEBUG
            if (!metadata_ || !(metadata_->index(i, j, k) < metadata_->size()))
                exit(-1);
            if (!this->is_set)
                exit(-2);
#endif
#endif
            return (this->m_fields[0])[metadata_->index(i, j, k)];
        }

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)

            This interface is not exposed to the user, it gets called from storage.hpp
        */
        GT_FUNCTION
        value_type const &access_data_impl(
            storage_info_type const *metadata_, uint_t const &i, uint_t const &j, uint_t const &k) const {

#ifdef __CUDA_ARCH__
            assert(metadata_ && metadata_->index(i, j, k) < metadata_->size());
            assert(this->is_set);
#else
#ifndef NDEBUG
            if (!metadata_ || !(metadata_->index(i, j, k) < metadata_->size()))
                exit(-1);
            if (!this->is_set)
                exit(-2);
#endif
#endif

            return (this->m_fields[0])[metadata_->index(i, j, k)];
        }

#endif // CXX11_ENABLED

      public:
        /**@brief swaps two arbitrary snapshots in two arbitrary data field dimensions

           @tparam SnapshotFrom one snapshot
           @tparam DimFrom one dimension
           @tparam SnapshotTo the second snapshot
           @tparam DimTo the second dimension

           syntax:
           swap<3,1>::with<4,1>::apply(storage_);
        */
        template < ushort_t SnapshotFrom, ushort_t DimFrom = 0 >
        struct swap {
            template < ushort_t SnapshotTo, ushort_t DimTo = 0 >
            struct with {

                template < typename Storage >
                GT_FUNCTION static void apply(Storage &storage_) {
                    super::template swap< SnapshotFrom, DimFrom >::template with< SnapshotTo, DimTo >::apply(storage_);
                    storage_.clone_to_device();
                }
            };
        };
    };

/**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each
   dimension has arbitrary size 'Number'.

   Annoyngly enough does not work with CUDA 6.5
*/
#if defined(CXX11_ENABLED)

    /** @brief syntactic sugar for defining a data field

        Given a storage type and the dimension number it generates the correct data field type
        @tparam Storage the basic storage used
        @tparam Number the number of snapshots in each dimension
     */
    template < class Storage, uint_t... Number >
    struct field_reversed;

    /**
     @brief specialization for the GPU storage
     the defined type is storage (which is clonable_to_gpu)
    */
    template < class BaseStorage, uint_t... Number >
    struct field_reversed< storage< BaseStorage >, Number... > {
        typedef storage< data_field< storage_list< base_storage< typename BaseStorage::pointer_type,
                                                       typename BaseStorage::storage_info_type,
                                                       accumulate(add_functor(), ((uint_t)Number)...) >,
            Number - 1 >... > > type;
    };

    /**
       @brief specialization for the CPU storage (base_storage)
       the difference being that the type is directly the base_storage (which is not clonable_to_gpu)
    */
    template < class PointerType, class MetaData, ushort_t FD, uint_t... Number >
    struct field_reversed< base_storage< PointerType, MetaData, FD >, Number... > {
        typedef data_field<
            storage_list< base_storage< PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number)...) >,
                Number - 1 >... > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, GPU storage)*/
    template < typename PointerType, typename MetaData, ushort_t FieldDimension, uint_t... Number >
    struct field_reversed< no_storage_type_yet< storage< base_storage< PointerType, MetaData, FieldDimension > > >,
        Number... > {
        typedef no_storage_type_yet< storage< data_field<
            storage_list< base_storage< PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number)...) >,
                Number - 1 >... > > > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, CPU storage)*/
    template < typename PointerType, typename MetaData, ushort_t FieldDimension, uint_t... Number >
    struct field_reversed< no_storage_type_yet< base_storage< PointerType, MetaData, FieldDimension > >, Number... > {
        typedef no_storage_type_yet< data_field<
            storage_list< base_storage< PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number)...) >,
                Number - 1 >... > > type;
    };

    /**@brief interface for definig a data field

       @tparam Storage the basic storage type shared by all the snapshots
       @tparam First  all the subsequent parameters define the dimensionality of the snapshot arrays
        in all the data field dimensions
     */
    template < class Storage, uint_t First, uint_t... Number >
    struct field {
        GRIDTOOLS_STATIC_ASSERT(is_storage<Storage>::value, "wrong type");
        typedef typename reverse_pack< Number... >::template apply< field_reversed, Storage, First >::type::type type;
    };

    template < class ... Storage, uint_t First, uint_t... Number >
    struct field<data_field<Storage ...>, First, Number...> {
        // GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), is_storage<Storage>::value ...), "wrong type");
        typedef typename reverse_pack< Number... >::template apply< field_reversed, typename data_field<Storage ...>::basic_type, First >::type::type type;
    };


#endif

    template < typename T >
    std::ostream &operator<<(std::ostream &s, storage< T > const &x) {
        s << "storage< " << static_cast< T const & >(x) << " > ";
        return s;
    }

    template < typename T >
    struct is_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_storage< storage< T > > : boost::mpl::true_ {};

#ifdef CXX11_ENABLED
    template < typename T >
    struct is_storage_list;

    template < typename T, uint_t U >
    struct is_storage< storage_list< T, U > > : boost::mpl::true_ {};

    template < typename... T >
    struct is_storage< data_field< T... > > : boost::mpl::true_ {};

#endif

    template < typename PointerType, typename MetaData, ushort_t FieldDimension >
    struct is_storage< base_storage< PointerType, MetaData, FieldDimension > > : boost::mpl::true_ {};

} // namespace gridtools
