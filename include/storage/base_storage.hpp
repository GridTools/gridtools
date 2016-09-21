/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include "../common/array.hpp"
#include "../common/pointer.hpp"
#include "../common/string_c.hpp"
#include "base_storage_impl.hpp"
#include "wrap_pointer.hpp"

/**@file
   @brief Implementation of the \ref gridtools::base_storage "main storage class", used by all backends, for temporary
   and non-temporary storage
*/

namespace gridtools {

    template < typename T >
    struct is_meta_storage;

    /***************************************/
    /************* base_storage ************/
    /***************************************/

    template < typename PointerType, typename MetaData, ushort_t FieldDimension = 1 >
    struct base_storage {
#ifdef CXX11_ENABLED
        template < typename PT, typename MD, ushort_t FD >
        using type_tt = base_storage< PT, MD, FD >;
#endif
        GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaData >::type::value, "wrong meta_storage type");
        typedef base_storage< PointerType, MetaData, FieldDimension > basic_type;
        typedef PointerType pointer_type;
        typedef typename pointer_type::pointee_t value_type;
        // consistency with STL
        typedef value_type *iterator;
        typedef value_type const *const_iterator;

        typedef MetaData storage_info_type;
        typedef typename MetaData::layout layout;
        static const bool is_temporary = storage_info_type::is_temporary;
        static const ushort_t n_width = 1;
        static const ushort_t space_dimensions = MetaData::space_dimensions;
        static const ushort_t field_dimensions = FieldDimension;
        // prohibit calls to copy ctor, copy assignment, and prohibit implicit conversions
      private:
        base_storage(const base_storage &);
        base_storage(base_storage &);
        base_storage &operator=(base_storage);
        base_storage &operator=(const base_storage &);

      protected:
        bool is_set;
        const char *m_name;
        array< pointer_type, field_dimensions > m_fields;
        pointer< const MetaData > m_meta_data;

      public:
        template < typename T, typename M, bool I, ushort_t F >
        friend std::ostream &operator<<(std::ostream &, base_storage< T, M, F > const &);

        /**
         * @brief the parallel storage calls the empty constructor to do lazy initialization
         */
        base_storage(
            MetaData const *meta_data_, char const *s = "default uninitialized storage", bool do_allocate = true)
            : is_set(false), m_name(malloc_and_copy(s)), m_meta_data(meta_data_) {
            if (do_allocate) {
                allocate();
            }
        }

        /**
         * @brief 3D storage constructor
         * @tparam FloatType is the floating point type passed to the constructor for initialization.
         * It is a template parameter in order to match float, double, etc...
         */
        base_storage(MetaData const *meta_data_, value_type const &init, char const *s = "default initialized storage")
            : is_set(false), m_name(malloc_and_copy(s)), m_meta_data(meta_data_) {
            allocate();
            assert(is_set && "allocation failed.");
            initialize(init, 1);
        }

        /**
         * @brief default constructor sets all the data members given the storage dimensions
         */
        template < typename Ret, typename T >
        base_storage(MetaData const *meta_data_,
            Ret (*func)(T const &, T const &, T const &),
            char const *s = "storage initialized with lambda")
            : is_set(false), m_name(malloc_and_copy(s)), m_meta_data(meta_data_) {
            allocate();
            assert(is_set && "allocation failed.");
            initialize(func, 1);
        }

        /**
         * @brief 3D constructor with the storage pointer provided externally
         *
         * This interface handles the case in which the storage is allocated from the python interface.
         * Since this storage gets freed inside python, it must be instantiated as a 'managed outside'
         * wrap_pointer. In this way the storage destructor will not free the pointer.
         */
        template < typename FloatType >
        explicit base_storage(MetaData const *meta_data_, FloatType *ptr, char const *s = "externally managed storage")
            : is_set(false), m_name(malloc_and_copy(s)), m_meta_data(meta_data_) {
            m_fields[0] = pointer_type(ptr, true);
            if (FieldDimension > 1) {
                allocate(FieldDimension, 1, true);
            }
        }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~base_storage() {
            delete[] m_name;
            release();
        }

        void h2d_update() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        void d2h_update() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].update_cpu();
        }

        void set_on_device() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].set_on_device();
        }

        void set_on_host() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].set_on_host();
        }

#ifdef CXX11_ENABLED
        /**
           explicitly disables the case in which the storage_info is passed by copy.
        */
        template < typename... T >
        base_storage(typename basic_type::storage_info_type &&, T...) = delete;
#endif

        /**@brief allocating memory for the data */
        void allocate(
            ushort_t const &dims = FieldDimension, ushort_t const &offset = 0, bool externally_managed = false) {
            assert(!is_set && "this storage is already allocated.");
            assert(dims > offset);
            assert(dims <= field_dimensions);
            is_set = true;
            for (ushort_t i = 0; i < dims; ++i) {
                m_fields[i + offset] = pointer_type(m_meta_data->size(), externally_managed);
            }
        }

        /**@brief releasing the pointers to the data, and deleting them in case they need to be deleted */
        void release() {
            if (is_set) {
                for (ushort_t i = 0; i < field_dimensions; ++i)
                    m_fields[i].free_it();
                is_set = false;
            }
        }

        /** @brief initializes with a constant value */
        GT_FUNCTION
        void initialize(value_type const &init, ushort_t const &dims = field_dimensions) {
            // if this fails you used the wrong constructor (i.e. the empty one)
            assert(is_set);
#ifdef _GT_RANDOM_INPUT
            srand(12345);
#endif
            for (ushort_t f = 0; f < dims; ++f) {
                for (uint_t i = 0; i < m_meta_data->size(); ++i) {
#ifdef _GT_RANDOM_INPUT
                    (m_fields[f])[i] = init * rand();
#else
                    (m_fields[f])[i] = init;
#endif
                }
            }
        }

        /** @brief initializes with a lambda function
                NOTE: valid for 3D storages only
         */
        template < typename Ret, typename T >
        GT_FUNCTION void initialize(
            Ret (*func)(T const &, T const &, T const &), ushort_t const &dims = field_dimensions) {
            GRIDTOOLS_STATIC_ASSERT(
                space_dimensions == 3, "this initialization is valid for storages with 3 space dimensions");
            // if this fails  you used the wrong constructor (i.e. the empty one)
            assert(is_set);
            assert(dims <= field_dimensions);

            for (ushort_t f = 0; f < dims; ++f)
                for (uint_t i = 0; i < m_meta_data->template dim< 0 >(); ++i)
                    for (uint_t j = 0; j < m_meta_data->template dim< 1 >(); ++j)
                        for (uint_t k = 0; k < m_meta_data->template dim< 2 >(); ++k)
                            (m_fields[f])[m_meta_data->index(i, j, k)] = func(i, j, k);
        }

        /**@brief sets the name of the current field*/
        GT_FUNCTION
        void set_name(char const *const &string) {
            if (m_name)
                delete[] m_name;
            m_name = malloc_and_copy(string);
        }

        /**@brief get the name of the current field*/
        GT_FUNCTION
        char const *get_name() const { return m_name; }

        static void text() { std::cout << BOOST_CURRENT_FUNCTION << std::endl; }

        /** @brief returns the last memory address of the data field */
        GT_FUNCTION
        const_iterator max_addr() const { return &((m_fields[field_dimensions - 1])[m_meta_data->size()]); }

        /** @brief returns (by reference) the value of the data field at the index "index_" */
        template < typename UInt >
        GT_FUNCTION value_type const &operator[](UInt const &index_) const {
            assert(index_ < m_meta_data->size());
            assert(is_set);
            GRIDTOOLS_STATIC_ASSERT(boost::is_integral< UInt >::value,
                "wrong type to the storage [] operator (the argument must be integral)");
            return (m_fields[0])[index_];
        }

#ifdef CXX11_ENABLED

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k) */
        template < typename... UInt >
        GT_FUNCTION value_type &operator()(UInt const &... dims) {
            assert(m_meta_data->index(dims...) < m_meta_data->size());
            assert(is_set);
            return (m_fields[0])[m_meta_data->index(dims...)];
        }

        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k) */
        template < typename... UInt >
        GT_FUNCTION value_type const &operator()(UInt const &... dims) const {
            assert(m_meta_data->index(dims...) < m_meta_data->size());
            assert(is_set);
            return (m_fields[0])[m_meta_data->index(dims...)];
        }
#else // CXX11_ENABLED

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION value_type &operator()(uint_t const &i, uint_t const &j, uint_t const &k) {
            assert(m_meta_data->index(i, j, k) < m_meta_data->size());
            assert(is_set);
            return (m_fields[0])[m_meta_data->index(i, j, k)];
        }

        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION value_type const &operator()(uint_t const &i, uint_t const &j, uint_t const &k) const {
            assert(m_meta_data->index(i, j, k) < m_meta_data->size());
            assert(is_set);
            return (m_fields[0])[m_meta_data->index(i, j, k)];
        }

#endif

        /**@brief prints the first values of the field to standard output*/
        void print() const { print(std::cout); }

        /**@brief prints a single value of the data field given the coordinates*/
        void print_value(uint_t i, uint_t j, uint_t k) {
            printf("value(%d, %d, %d)=%f, at index %d on the data\n",
                i,
                j,
                k,
                (m_fields[0])[m_meta_data->index(i, j, k)],
                m_meta_data->index(i, j, k));
        }

        static const std::string info_string;

        /**@brief printing a portion of the content of the data field*/
        template < typename Stream >
        void print(Stream &stream, uint_t t = 0) const {
            stream << "| j" << std::endl;
            stream << "| j" << std::endl;
            stream << "v j" << std::endl;
            stream << "---> k" << std::endl;

            ushort_t MI = 12;
            ushort_t MJ = 12;
            ushort_t MK = 12;
            for (uint_t i = 0; i < m_meta_data->template dim< 0 >();
                 i += std::max((uint_t)1, m_meta_data->template dim< 0 >() / MI)) {
                for (uint_t j = 0; j < m_meta_data->template dim< 1 >();
                     j += std::max((uint_t)1, m_meta_data->template dim< 1 >() / MJ)) {
                    for (uint_t k = 0; k < m_meta_data->template dim< 2 >();
                         k += std::max((uint_t)1, m_meta_data->template dim< 1 >() / MK)) {
                        stream << "["
                               // << i << ","
                               // << j << ","
                               // << k << ")"
                               << (m_fields[t])[m_meta_data->index(i, j, k)] << "] ";
                    }
                    stream << std::endl;
                }
                stream << std::endl;
            }
            stream << std::endl;
        }

        /**@brief returns the data field*/
        GT_FUNCTION
        pointer_type const &data() const { return (m_fields[0]); }

        /** @brief returns a const pointer to the data field*/
        GT_FUNCTION
        pointer_type const *fields() const { return &(m_fields[0]); }

        /** @brief returns a const pointer to the data field*/
        template < typename ID >
        GT_FUNCTION value_type *access_value() const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < field_dimensions),
                "Error: trying to access a field storage index beyond the field dimensions");
            return fields()[ID::value].get();
        }

        /** @brief returns a non const pointer to the data field*/
        GT_FUNCTION
        pointer_type *fields_view() { return &(m_fields[0]); }

        /** @brief returns a const ref to the meta data field*/
        GT_FUNCTION
        pointer< const storage_info_type > meta_data() const { return m_meta_data; }

        GT_FUNCTION
        void set_meta_data(const storage_info_type *st) { m_meta_data = st; }
        /**
           @brief API for compatibility with backends other than host
           avoids the introduction of #ifdefs
         */
        void clone_to_device() {}

        GT_FUNCTION
        void set_externally_managed(bool val_) {
            for (ushort_t i = 0; i < field_dimensions; ++i) {
                m_fields[i].set_externally_managed(val_);
            }
        }

        GT_FUNCTION
        void unset() { is_set = false; }
    };

    /** \addtogroup specializations Specializations
            Partial specializations
            @{
    */
    template < typename PointerType, typename MetaData, ushort_t Dim >
    const std::string base_storage< PointerType, MetaData, Dim >::info_string = boost::lexical_cast< std::string >(
        "-1");

    template < typename PointerType, typename MetaData, ushort_t Dim >
    const ushort_t base_storage< PointerType, MetaData, Dim >::field_dimensions;

    template < typename PointerType, typename MetaData, ushort_t Dim >
    struct is_temporary_storage< base_storage< PointerType, MetaData, Dim > *& >
        : boost::mpl::bool_< MetaData::is_temporary > {};

    template < typename PointerType, typename MetaData, ushort_t Dim >
    struct is_temporary_storage< base_storage< PointerType, MetaData, Dim > * >
        : boost::mpl::bool_< MetaData::is_temporary > {};

    template < typename PointerType, typename MetaData, ushort_t Dim >
    struct is_temporary_storage< base_storage< PointerType, MetaData, Dim > >
        : boost::mpl::bool_< MetaData::is_temporary > {};

    template < template < typename T > class Decorator, typename BaseType >
    struct is_temporary_storage< Decorator< BaseType > > : is_temporary_storage< BaseType > {};
    template < template < typename T > class Decorator, typename BaseType >
    struct is_temporary_storage< Decorator< BaseType > * > : is_temporary_storage< BaseType * > {};
    template < template < typename T > class Decorator, typename BaseType >
    struct is_temporary_storage< Decorator< BaseType > & > : is_temporary_storage< BaseType & > {};
    template < template < typename T > class Decorator, typename BaseType >
    struct is_temporary_storage< Decorator< BaseType > *& > : is_temporary_storage< BaseType *& > {};

#ifdef CXX11_ENABLED
    // Decorator is the storage class
    template < template < typename... T > class Decorator, typename First, typename... BaseType >
    struct is_temporary_storage< Decorator< First, BaseType... > >
        : is_temporary_storage< typename First::basic_type > {};

    // Decorator is the storage class
    template < template < typename... T > class Decorator, typename First, typename... BaseType >
    struct is_temporary_storage< Decorator< First, BaseType... > * >
        : is_temporary_storage< typename First::basic_type * > {};

    // Decorator is the storage class
    template < template < typename... T > class Decorator, typename First, typename... BaseType >
    struct is_temporary_storage< Decorator< First, BaseType... > & >
        : is_temporary_storage< typename First::basic_type & > {};

    // Decorator is the storage class
    template < template < typename... T > class Decorator, typename First, typename... BaseType >
    struct is_temporary_storage< Decorator< First, BaseType... > *& >
        : is_temporary_storage< typename First::basic_type *& > {};
#else
    // Decorator is the storage class
    template < template < typename T1, typename T2, typename T3 > class Decorator,
        typename First,
        typename B2,
        typename B3 >
    struct is_temporary_storage< Decorator< First, B2, B3 > > : is_temporary_storage< typename First::basic_type > {};

    // Decorator is the storage class
    template < template < typename T1, typename T2, typename T3 > class Decorator,
        typename First,
        typename B2,
        typename B3 >
    struct is_temporary_storage< Decorator< First, B2, B3 > * > : is_temporary_storage< typename First::basic_type * > {
    };

    // Decorator is the storage class
    template < template < typename T1, typename T2, typename T3 > class Decorator,
        typename First,
        typename B2,
        typename B3 >
    struct is_temporary_storage< Decorator< First, B2, B3 > & > : is_temporary_storage< typename First::basic_type & > {
    };

    // Decorator is the storage class
    template < template < typename T1, typename T2, typename T3 > class Decorator,
        typename First,
        typename B2,
        typename B3 >
    struct is_temporary_storage< Decorator< First, B2, B3 > *& >
        : is_temporary_storage< typename First::basic_type *& > {};

#endif // CXX11_ENABLED
    /**@}*/
    template < typename T, typename U, bool B, ushort_t D >
    std::ostream &operator<<(std::ostream &s, base_storage< T, U, D > const &x) {
        s << "base_storage <T,U,"
          << " " << D << "> ";
        s << x.m_dims[0] << ", " << x.m_dims[1] << ", " << x.m_dims[2] << ". ";
        return s;
    }

    template < typename T >
    struct is_storage : boost::mpl::false_ {};

    template < typename T, typename V, ushort_t D >
    struct is_storage< base_storage< T, V, D > > : boost::mpl::true_ {};

} // namespace gridtools
