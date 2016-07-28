/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <iosfwd>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

#include "data_field.hpp"
#ifdef _USE_GPU_
#include "hybrid_pointer.hpp"
#endif
#include "wrap_pointer.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/reverse_pack.hpp"
#include "../common/pointer_metafunctions.hpp"
#endif
#include "meta_storage_extender.hpp"

/**
@file
@brief Host-side storage class. This class contains a hybrid- or wrap_pointer to the base_storage.
The base storage can be copied to the device and back.
*/

namespace gridtools {

    /***************************************/
    /********* no_storage_type_yet *********/
    /***************************************/

    template < typename RegularMetaStorageType >
    struct no_meta_storage_type_yet;

    /**
    * @brief Type to indicate that the type is not decided yet
    */
    template < typename RegularStorageType >
    struct no_storage_type_yet {
#ifdef CXX11_ENABLED
        template < typename PT, typename MD, ushort_t FD >
        using type_tt = typename RegularStorageType::template type_tt< PT, MD, FD >;
#endif
        typedef typename RegularStorageType::basic_type type;
        typedef typename type::storage_info_type storage_info_type;
        typedef typename type::layout layout;
        typedef typename type::const_iterator const_iterator;
        typedef typename type::basic_type basic_type;
        typedef typename type::pointer_type pointer_type;
        static const ushort_t n_width = basic_type::n_width;
        static const ushort_t field_dimensions = basic_type::field_dimensions;
        typedef void storage_type;
        typedef typename type::iterator iterator;
        typedef typename type::value_type value_type;
        typedef typename RegularStorageType::storage_ptr_t storage_ptr_t;
        static const ushort_t space_dimensions = RegularStorageType::space_dimensions;
        static const bool is_temporary = RegularStorageType::is_temporary;
        static void text() { std::cout << "text: no_storage_type_yet<" << RegularStorageType() << ">" << std::endl; }
        storage_ptr_t storage_pointer() const { assert(false); }
        void clone_to_device() { assert(false); }
        void set_on_device() { assert(false); }
        void d2h_update() { assert(false); }
        pointer< const storage_ptr_t > get_storage_pointer() const { assert(false); }
        void info(std::ostream &out_s) const {
            out_s << "No sorage type yet for storage type " << RegularStorageType() << "\n";
        }
    };

    template < typename T >
    struct is_no_storage_type_yet : boost::mpl::false_ {};

    template < typename RegularStorageType >
    struct is_no_storage_type_yet< no_storage_type_yet< RegularStorageType > > : boost::mpl::true_ {};

    /**
    *  @brief stream operator, for debugging purpose
    */
    template < typename RST >
    std::ostream &operator<<(std::ostream &s, no_storage_type_yet< RST >) {
        return s << "no_storage_type_yet<" << RST() << ">";
    }

    /***************************************/
    /*************** storage ***************/
    /***************************************/

    template < typename BaseStorage >
    struct storage {
// some forwarding and convenience typedefs
#ifdef CXX11_ENABLED
        template < typename PT, typename MD, ushort_t FD >
        using type_tt = typename BaseStorage::template type_tt< PT, MD, FD >;
#endif
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef storage< BaseStorage > this_type;
        typedef typename BaseStorage::storage_info_type storage_info_type;
        typedef typename BaseStorage::iterator iterator;
        typedef typename BaseStorage::value_type value_type;
        typedef typename BaseStorage::pointer_type pointer_type;
        static const bool is_temporary = BaseStorage::is_temporary;
        const static ushort_t field_dimensions = BaseStorage::field_dimensions;
        const static ushort_t space_dimensions = BaseStorage::space_dimensions;
        const static ushort_t n_width = BaseStorage::n_width;
        // get the right pointer type to keep the base storage
        typedef typename boost::mpl::if_< is_wrap_pointer< pointer_type >,
            wrap_pointer< BaseStorage, false >,
            typename boost::mpl::if_< is_hybrid_pointer< pointer_type >,
                                              hybrid_pointer< BaseStorage, false >,
                                              boost::mpl::void_ >::type >::type storage_ptr_t;
        // get the right pointer type to keep the meta data
        typedef typename boost::mpl::if_< is_wrap_pointer< pointer_type >,
            wrap_pointer< const storage_info_type, false >,
            typename boost::mpl::if_< is_hybrid_pointer< pointer_type >,
                                              hybrid_pointer< const storage_info_type, false >,
                                              boost::mpl::void_ >::type >::type meta_data_ptr_t;

      protected:
        meta_data_ptr_t m_meta_data;
        storage_ptr_t m_storage;
        bool m_on_host;
        template < typename T >
        storage(T);

      public:
        bool is_on_host() const { return m_on_host; }

        void clone_to_device() {
#ifdef _USE_GPU_
            // if (!m_on_host)
            //     return;
            // clone meta dato to device
            m_meta_data.update_gpu(); // useless if meta data is constexpr
            // set the new meta data pointer in the storage
            m_storage.set_on_host();
            m_storage.get_cpu_p()->set_meta_data(m_meta_data.get_pointer_to_use());
            // m_storage.set_on_host();
            // m_storage.get_cpu_p()->set_meta_data(m_meta_data.get_pointer_to_use());
            // update the storage itself
            m_storage.update_gpu();
#endif
        }

        /** @brief clone storage + contents to gpu */
        void d2h_update() {
            if (m_on_host)
                return;
            // no need to copy result back, just switch the pointers
            m_meta_data.set_on_host();
            // clone the storage itself from device
            // m_storage.update_cpu();
            // no need to copy result back, just switch the pointers
            m_storage.set_on_host();
            // set the new meta data pointer in the storage
            (*m_storage).set_meta_data(m_meta_data.get_pointer_to_use());
            // clone storage contents from device
            (*m_storage).d2h_update();
            m_on_host = true;
        }

        /** @brief clone storage + contents from gpu */
        void h2d_update() {
            if (!m_on_host)
                return;
            // clone meta dato to device
            m_storage.set_on_host();
            m_meta_data.update_gpu();
            // m_meta_data.set_on_host();
            // set the new meta data pointer in the storage
            (*m_storage).set_meta_data(m_meta_data.get_gpu_p());
            // clone storage contents to device
            (*m_storage).h2d_update();
            // clone the storage itself to device
            m_storage.update_gpu();
            // set m_on_host to false
            m_on_host = false;
        }

        /* Following method are just forwarding methods to the base_storage. */
        storage_info_type const &meta_data() const {
            assert(m_on_host);
            return *m_meta_data;
        }

        pointer< storage_info_type const > get_meta_data_pointer() const {
            return pointer< storage_info_type const >(m_meta_data.get_pointer_to_use());
        }

        pointer< storage_info_type const > get_meta_data_pointer() {
            return pointer< storage_info_type const >(m_meta_data.get_pointer_to_use());
        }

        pointer_type const &data() const {
            assert(m_on_host);
            return (*m_storage).data();
        }

        pointer_type const *fields() const {
            assert(m_on_host);
            return (*m_storage).fields();
        }

        template < typename ID >
        value_type *access_value() const {
            assert(m_on_host);
            return (*m_storage).template access_value< ID >();
        }

        GT_FUNCTION
        pointer_type *fields_view() {
            assert(m_on_host);
            return (*m_storage).fields_view();
        }

        void initialize(value_type const &init, ushort_t const &dims = BaseStorage::field_dimensions) {
            assert(m_on_host);
            (*m_storage).initialize(init, dims);
        }

        void initialize(uint_t (*func)(uint_t const &, uint_t const &, uint_t const &),
            ushort_t const &dims = BaseStorage::field_dimensions) {
            assert(m_on_host);
            assert(func);
            (*m_storage).initialize(func, dims);
        }

        void allocate(ushort_t const &dims = BaseStorage::field_dimensions, ushort_t const &offset = 0) {
            assert(m_on_host);
            (*m_storage).allocate(dims, offset);
        }

        template <ushort_t snapshot_, ushort_t dim_>
        base_storage<pointer_type, typename basic_type::storage_info_type, 1> get_storage( ) {
            GRIDTOOLS_STATIC_ASSERT(is_data_field<BaseStorage>::value, "error");
            return m_storage->template get_storage<snapshot_, dim_>();
        }

        GT_FUNCTION
        pointer< storage_ptr_t > get_storage_pointer() { return pointer< storage_ptr_t >(&m_storage); }

        GT_FUNCTION
        storage_ptr_t storage_pointer() { return m_storage; }

        GT_FUNCTION
        pointer< const storage_ptr_t > get_storage_pointer() const {
            return pointer< const storage_ptr_t >(&m_storage);
        }

        void print() {
            assert(m_on_host);
            (*m_storage).print();
        }

        template < typename T >
        void print(T &s) {
            assert(m_on_host);
            (*m_storage).print(s);
        }

        GT_FUNCTION
        void print_value(uint_t i, uint_t j, uint_t k) {
            assert(m_on_host);
            (*m_storage).print_value(i, j, k);
        }

        GT_FUNCTION
        char const *get_name() const {
            assert(m_on_host);
            return (*m_storage).get_name();
        }

        GT_FUNCTION
        void set_name(char const *const &string) {
            assert(m_on_host);
            (*m_storage).set_name(string);
        }

#if defined(CXX11_ENABLED)

        /** @brief method to return an element of a specific storage in a data field of storages

            \tparam Snapshot the index of the snapshot in the current storage list
            \tparam FieldDim the index of the storage list (there is one storage_list per field dimension)
            \param args the indices used to access the storage, once it has been identified by the Snaphot and FieldDim
           coordinates
         */
        template < short_t Snapshot = 0, short_t FieldDim = 0, typename... Int >
        GT_FUNCTION value_type &get_value(Int... args) {
            GRIDTOOLS_STATIC_ASSERT(
                is_data_field< super >::value, "the get_value API is available only for data field storages.");
            return (*m_storage).template get_value< Snapshot, FieldDim, Int... >(args...);
        }

        /** @brief method to return an element of a specific storage in a data field of storages

            \tparam Snapshot the index of the snapshot in the current storage list
            \tparam FieldDim the index of the storage list (there is one storage_list per field dimension)
            \param args the indices used to access the storage, once it has been identified by the Snaphot and FieldDim
           coordinates
         */
        template < short_t Snapshot = 0, short_t FieldDim = 0, typename... Int >
        GT_FUNCTION value_type const &get_value(Int... args) const {
            GRIDTOOLS_STATIC_ASSERT(
                is_data_field< super >::value, "the get_value API is available only for data field storages.");
            return (*m_storage).template get_value< Snapshot, FieldDim, Int... >(args...);
        }

        /** @brief method to return the pointer to a specific storage in a data field of storages

            \tparam Snapshot the index of the snapshot in the current storage list
            \tparam FieldDim the index of the storage list (there is one storage_list per field dimension)
         */
        template < short_t snapshot = 0, short_t field_dim = 0 >
        GT_FUNCTION pointer_type const &get() const {
            GRIDTOOLS_STATIC_ASSERT(
                is_data_field< super >::value, "the get_value API is available only for data field storages.");
            return (*m_storage)
                .fields_view()[_impl::access< super::n_width - (field_dim), typename super::traits >::type::n_fields +
                               snapshot];
        }

        /** @brief method to return the pointer to a specific storage in a data field of storages

            \tparam Snapshot the index of the snapshot in the current storage list
            \tparam FieldDim the index of the storage list (there is one storage_list per field dimension)
         */
        template < short_t Snapshot = 0, short_t FieldDim = 0 >
        GT_FUNCTION pointer_type &get() {
            GRIDTOOLS_STATIC_ASSERT(
                is_data_field< super >::value, "the get_value API is available only for data field storages.");
            return (*m_storage)
                .fields_view()[_impl::access< super::n_width - (FieldDim), typename super::traits >::type::n_fields +
                               Snapshot];
        }

        /** @brief method to set the pointer to a specific storage in a data field of storages

            \tparam Snapshot the index of the snapshot in the current storage list
            \tparam FieldDim the index of the storage list (there is one storage_list per field dimension)
            \tparam F can be several thongs, e.g. a pointer to a storage, an initialization value value, or a lambda
         */
        template < short_t Snapshot = 0, short_t FieldDim = 0, typename F >
        GT_FUNCTION void set(F f) {
            GRIDTOOLS_STATIC_ASSERT(
                is_data_field< super >::value, "the get_value API is available only for data field storages.");
            // F is protected inside data_field
            assert(m_on_host);
            (*m_storage).set< Snapshot, FieldDim >(f);
        }

        // forwarding constructor
        template < class... ExtraArgs >
        explicit storage(storage_info_type const &meta_data_, ExtraArgs const &... args)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), args...), false), m_on_host(true) {}
#else // CXX11_ENABLED

        explicit storage(storage_info_type const &meta_data_, const char* name_, bool do_allocate)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), name_, do_allocate), false), m_on_host(true) {}

        explicit storage(storage_info_type const &meta_data_, value_type const &init)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), init, "default storage"), false), m_on_host(true) {}

        explicit storage(
            storage_info_type const &meta_data_, value_type const &init, const char *name)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), init, name), false),
              m_on_host(true) {}

        template < typename Ret, typename T >
        explicit storage(storage_info_type const &meta_data_, Ret (*func)(T const &, T const &, T const &))
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), func), false), m_on_host(true) {}

        template < class FloatType >
        explicit storage(storage_info_type const &meta_data_, FloatType *arg)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), (FloatType *)arg), false), m_on_host(true) {}

        template < class FloatType >
        explicit storage(storage_info_type const &meta_data_, FloatType *arg, const char *name)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use(), (FloatType *)arg, name), false),
              m_on_host(true) {}

#endif // CXX11_ENABLED

        /** @brief destructor freeing up the pointers to the storages
         */
        ~storage() {
            m_storage.free_it();
            m_meta_data.free_it();
        }

        /**@brief releasing the pointers to the data, and deleting them in case they need to be deleted */
        void release() {
            assert(m_on_host);
            (*m_storage).release();
        }

        GT_FUNCTION
        BaseStorage *get_pointer_to_use() { return m_storage.get_pointer_to_use(); }

        explicit storage(storage_info_type const &meta_data_)
            : m_meta_data(new storage_info_type(meta_data_), false),
              m_storage(new BaseStorage(m_meta_data.get_pointer_to_use()), false), m_on_host(true) {}

        template < typename UInt >
        GT_FUNCTION value_type const &operator[](UInt const &index_) const {
            assert(m_on_host && "The accessed storage was not copied back from the device yet.");
            return (*m_storage)[index_];
        }

#ifdef CXX11_ENABLED

        /**
         * explicitly disables the case in which the storage_info is passed as r- or x-value.
         */
        template < typename... T >
        storage(storage_info_type &&, T...) = delete;

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)
         *  this api is callable from the host only. The function that is used to .
         */
        template < typename... UInt >
        GT_FUNCTION value_type &operator()(UInt... dims) {
            assert(m_on_host && "The accessed storage was not copied back from the device yet.");
            return (*m_storage)(dims...);
        }

        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k)
         *  this api is callable from the host only. The function that is used to .
         */
        template < typename... UInt >
        GT_FUNCTION value_type const &operator()(UInt const &... dims) const {
            assert(m_on_host && "The accessed storage was not copied back from the device yet.");
            return (*m_storage)(dims...);
        }

#else // CXX11_ENABLED

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k)
         *  this api is callable from the host only. The function that is used to .
         */
        GT_FUNCTION
        value_type &operator()(uint_t const &i, uint_t const &j, uint_t const &k) {
            assert(m_on_host && "The accessed storage was not copied back from the device yet.");
            return (*m_storage)(i, j, k);
        }

        GT_FUNCTION
        const value_type &operator()(uint_t const &i, uint_t const &j, uint_t const &k) const {
            assert(m_on_host && "The accessed storage was not copied back from the device yet.");
            return (*m_storage)(i, j, k);
        }
#endif

        GT_FUNCTION
        void set_on_device() { m_on_host = false; }

        GT_FUNCTION
        void set_on_host() { m_on_host = true; }

        GT_FUNCTION
        void set_externally_managed(bool val_) { m_storage->set_externally_managed(val_); }

    }; // closing struct storage

/**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each
   dimension has arbitrary size 'Number'.

   Annoyngly enough does not work with CUDA 6.5
*/
#if defined(CXX11_ENABLED)

    template < typename T >
    struct is_any_storage;

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
    template < class Storage, uint_t... Number >
    struct field_reversed< storage< Storage >, Number... > {
        typedef storage< data_field< storage_list< base_storage< typename Storage::pointer_type,
                                                       typename Storage::storage_info_type,
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

    /**@brief interface for defining a data field

       @tparam Storage the basic storage type shared by all the snapshots
       @tparam First  all the subsequent parameters define the dimensionality of the snapshot arrays
            in all the data field dimensions
     */
    template < class Storage, uint_t First, uint_t... Number >
    struct field {
        GRIDTOOLS_STATIC_ASSERT(is_any_storage< Storage >::value, "wrong type");
        typedef typename reverse_pack< Number... >::template apply< field_reversed, Storage, First >::type::type type;
    };

    template < class... Storage, uint_t First, uint_t... Number >
    struct field< data_field< Storage... >, First, Number... > {
        // GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), is_storage<Storage>::value ...), "wrong type");
        typedef typename reverse_pack< Number... >::template apply< field_reversed,
            typename data_field< Storage... >::basic_type,
            First >::type::type type;
    };

#endif

    template < typename T >
    std::ostream &operator<<(std::ostream &s, storage< T > const &x) {
        s << "storage< " << static_cast< T const & >(x) << " > ";
        return s;
    }

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

} // namespace gridtools
