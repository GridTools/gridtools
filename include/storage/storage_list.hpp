#pragma once
#include "base_storage.hpp"
#ifdef CXX11_ENABLED
namespace gridtools {
    /** @brief storage class containing a buffer of data snapshots

        it is a list of \ref gridtools::base_storage "storages"

        \include storage.dox

    */
    template < typename Storage, short_t ExtraWidth >
    struct storage_list : public Storage {

        template < typename PT, typename MD, ushort_t FD >
        using type_tt = storage_list< typename Storage::template type_tt< PT, MD, FD >, ExtraWidth >;

        typedef storage_list< Storage, ExtraWidth > type;
        /*If the following assertion fails, you probably set one field dimension to contain zero (or negative)
         * snapshots. Each field dimension must contain one or more snapshots.*/
        GRIDTOOLS_STATIC_ASSERT(ExtraWidth > 0,
            "you probably set one field dimension to contain zero (or negative) "
            "snapshots. Each field dimension must contain one or more snapshots.");
        typedef Storage super;
        typedef typename super::pointer_type pointer_type;

        typedef typename super::basic_type basic_type;
        // typedef typename super::original_storage original_storage;
        typedef typename super::iterator_type iterator_type;
        typedef typename super::value_type value_type;

        /**@brief constructor*/
        template < typename... Args >
        storage_list(typename basic_type::storage_info_type const &meta_data_, Args const &... args_)
            : super(meta_data_, args_...) {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list() {}

        /**@brief device copy constructor*/
        template < typename T >
        __device__ storage_list(T const &other)
            : super(other) {
            // GRIDTOOLS_STATIC_ASSERT(n_width==T::n_width, "Dimension analysis error: copying two vectors with
            // different dimensions");
        }

        /**@brief printing the first values of all the snapshots contained in the discrete field*/
        void print() { print(std::cout); }

        /**@brief printing the first values of all the snapshots contained in the discrete field, given the output
         * stream*/
        template < typename Stream >
        void print(Stream &stream) {
            for (ushort_t t = 0; t < super::field_dimensions; ++t) {
                stream << " Component: " << t + 1 << std::endl;
                basic_type::print(stream, t);
            }
        }

        static const ushort_t n_width = ExtraWidth + 1;
    };

    /**@brief specialization: if the width extension is 0 we fall back on the base storage*/
    template < typename Storage >
    struct storage_list< Storage, 0 > : public Storage {
        template < typename PT, typename MD, ushort_t FD >
        using type_tt = storage_list< typename Storage::template type_tt< PT, MD, FD >, 0 >;

        typedef typename Storage::basic_type basic_type;
        typedef Storage super;

        // default constructor
        template < typename... Args >
        storage_list(typename basic_type::storage_info_type const &meta_data_, Args const &... args_)
            : super(meta_data_, args_...) {}

        // default constructor
        storage_list(typename basic_type::storage_info_type const &meta_data_) : super(meta_data_) {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list() {}

        /**dimension number of snaphsots for the current field dimension*/
        static const ushort_t n_width = Storage::n_width;

        /**@brief device copy constructor*/
        template < typename T >
        __device__ storage_list(T const &other)
            : super(other) {}
    };

    template < typename T >
    struct is_storage_list : boost::mpl::false_ {};

    template < typename Storage, short_t ExtraWidth >
    struct is_storage_list< storage_list< Storage, ExtraWidth > > : boost::mpl::true_ {};

} // namespace gridtools
#endif
