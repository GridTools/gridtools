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

    // template < typename T >
//     struct is_storage;

//     template < typename RegularMetaStorageType >
//     struct no_meta_storage_type_yet;

//     /**
//      * @brief Type to indicate that the type is not decided yet
//      */
//     template < typename RegularStorageType >
//     struct no_storage_type_yet {

// #ifdef CXX11_ENABLED
//         template < typename PT, typename MD, ushort_t FD >
//         using type_tt = typename RegularStorageType::template type_tt< PT, MD, FD >;
// #endif

//         typedef RegularStorageType type;
//         typedef no_meta_storage_type_yet< typename RegularStorageType::storage_info_type > storage_info_type;
//         typedef typename type::layout layout;
//         typedef typename type::const_iterator_type const_iterator_type;
//         typedef typename type::basic_type basic_type;
//         typedef typename type::pointer_type pointer_type;
//         static const ushort_t n_width = basic_type::n_width;
//         static const ushort_t field_dimensions = basic_type::field_dimensions;
//         typedef void storage_type;
//         typedef typename type::iterator_type iterator_type;
//         typedef typename type::value_type value_type;

//         // consistency with STL:
//         typedef iterator_type iterator;
//         typedef const_iterator_type const_iterator;

//         static const ushort_t space_dimensions = RegularStorageType::space_dimensions;
//         static const bool is_temporary = RegularStorageType::is_temporary;
//         static void text() { std::cout << "text: no_storage_type_yet<" << RegularStorageType() << ">" << std::endl; }
//         // std::string name() {return std::string("no_storage_yet NAMEname");}
//         void info() const { std::cout << "No sorage type yet for storage type " << RegularStorageType() << std::endl; }
//         typename RegularStorageType::storage_info_type const &meta_data() { assert(false); }
//         void set_on_device() { assert(false); }
//         void d2h_update() { assert(false); }
//         typename RegularStorageType::pointer_type *fields() { assert(false); }

// #ifdef CXX11_ENABLED
//         template < typename... Args >
//         void set(Args...) {
//             assert(false);
//         }
// #endif

//         void clone_to_device() { assert(false); }
//     };

//     template < typename T >
//     struct is_no_storage_type_yet : boost::mpl::false_ {};

//     template < typename RegularStorageType >
//     struct is_no_storage_type_yet< no_storage_type_yet< RegularStorageType > > : boost::mpl::true_ {};

//     /**
//        @brief stream operator, for debugging purpose
//     */
//     template < typename RST >
//     std::ostream &operator<<(std::ostream &s, no_storage_type_yet< RST >) {
//         return s << "no_storage_type_yet<" << RST() << ">";
//     }

//     /**
//        \anchor descr_storage
//        @brief main class for the basic storage

//        We define here an important naming convention. We call:

//        - the storages (or storage snapshots): are contiguous chunks of memory, accessed by 3 (by default, but not
// necessarily) indexes.
//        These structures are univocally defined by 3 (by default) integers. These are currently 2 strides and the total
// size of the chunks. Note that (in 3D) the relation between these quantities
//        (\f$stride_1\f$, \f$stride_2\f$ and \f$size\f$) and the dimensions x, y and z can be (depending on the storage
// layout chosen)
//        \f[
//        size=x*y*z \;;\;
//        stride_2=x*y \;;\;
//        stride_1=x .
//        \f]
//        The quantities \f$size\f$, \f$stride_2\f$ and \f$stride_1\f$ are arranged respectively in m_strides[0],
// m_strides[1], m_strides[2].
//        - the \ref gridtools::storage_list "storage list": is a list of pointers (or snapshots) to storages. The
// snapshots are arranged on a 1D array. The \ref gridtools::accessor "accessor" class is
//        responsible of computing the correct offests (relative to the given dimension) and address the storages
// correctly.
//        - the \ref gridtools::data_field "data field": is a collection of storage lists, and can contain one or more
// storage lists of different sizes. It can be seen as a vector of vectors of storage pointers.
//        (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)

//        The base_storage class has a 1-1 relation with the storage concept, while the subclasses extend the concept of
// storage to the structure represented in the ASCII picture below.

//        NOTE: the constraint of the snapshots accessed by the same data field are the following:
//        - the memory layout (strides, space dimensions) is one for all the snapshots, and all the snapshots
//        share the same iteration point
// \verbatim
// ############### 2D Storage ################
// #                    ___________\         #
// #                      time     /         #
// #                  | |*|*|*|*|*|*|        #
// # space, pressure  | |*|*|*|              #
// #    energy,...    v |*|*|*|*|*|          #
// #                                         #
// #                     ^ ^ ^ ^ ^ ^         #
// #                     | | | | | |         #
// #                      snapshots          #
// #                                         #
// ############### 2D Storage ################
// \endverbatim

//        The final storage which is effectly instantiated must be "clonable to the GPU", i.e. it must derive from the
// clonable_to_gpu struct.
//        This is achieved by using multiple inheritance.

//        NOTE CUDA: It is important when subclassing from a storage object to reimplement the __device__ copy constructor,
// and possibly the method 'copy_data_to_gpu' which are used when cloning the class to the CUDA device.

//        The base_storage class contains one snapshot. It univocally defines
//        the access pattern with three integers: the total storage sizes and
//        the two strides different from one.
//     */

//     template < typename T >
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
            is_set = true;
        }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~base_storage() {
            delete[] m_name;
            for (ushort_t i = 0; i < field_dimensions; ++i)
                m_fields[i].free_it();
        }

        void h2d_update() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        void d2h_update() {
            for (uint_t i = 0; i < field_dimensions; ++i)
                m_fields[i].update_cpu();
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
            for (ushort_t i = 0; i < field_dimensions; ++i)
                m_fields[i].free_it();
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
                for (uint_t i = 0; i < m_meta_data->template dims< 0 >(); ++i)
                    for (uint_t j = 0; j < m_meta_data->template dims< 1 >(); ++j)
                        for (uint_t k = 0; k < m_meta_data->template dims< 2 >(); ++k)
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

        // /** @brief update the GPU pointer */
        // void h2d_update() {
        //     for (uint_t i = 0; i < field_dimensions; ++i)
        //         m_fields[i].update_gpu();
        // }

        // void set_on_device() {
        //     for (uint_t i = 0; i < field_dimensions; ++i)
        //         m_fields[i].set_on_device();
        // }

        // /** @brief updates the CPU pointer */
        // void d2h_update() {
        //     for (uint_t i = 0; i < field_dimensions; ++i)
        //         m_fields[i].update_cpu();
        // }

        // /** @brief returns the last memry address of the data field */
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
            for (uint_t i = 0; i < m_meta_data->template dims< 0 >();
                 i += std::max((uint_t)1, m_meta_data->template dims< 0 >() / MI)) {
                for (uint_t j = 0; j < m_meta_data->template dims< 1 >();
                     j += std::max((uint_t)1, m_meta_data->template dims< 1 >() / MJ)) {
                    for (uint_t k = 0; k < m_meta_data->template dims< 2 >();
                         k += std::max((uint_t)1, m_meta_data->template dims< 1 >() / MK)) {
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
