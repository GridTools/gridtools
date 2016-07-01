/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used
   in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.
*/

#pragma once

#include <iosfwd>
#include "storage/storage_metafunctions.hpp"
#include "stencil-composition/arg_metafunctions_fwd.hpp"

namespace gridtools {

    // fwd decl
    template < typename T >
    struct is_arg;

    /** @brief binding between the placeholder (\tparam ArgType) and the storage (\tparam Storage)*/
    template < typename ArgType, typename Storage >
    struct arg_storage_pair {

        GRIDTOOLS_STATIC_ASSERT(is_arg< ArgType >::value, "wrong type");

      private:
        // arg_storage_pair(arg_storage_pair const&);
        arg_storage_pair();

      public:
        pointer< Storage > ptr;

        arg_storage_pair(arg_storage_pair const &other) : ptr(other.ptr) { assert(ptr.get()); }

        typedef ArgType arg_type;
        typedef Storage storage_type;

        arg_storage_pair(pointer< Storage > p) : ptr(p) {}

        arg_storage_pair(Storage *p) : ptr(p) {}
    };

    template < typename T >
    struct is_arg_storage_pair : boost::mpl::false_ {};

    template < typename ArgType, typename Storage >
    struct is_arg_storage_pair< arg_storage_pair< ArgType, Storage > > : boost::mpl::true_ {};

    /**
     * Type to create placeholders for data fields.
     *
     * There is a specialization for the case in which T is a temporary.
     * The default version applies to all the storage classes (including
     * user-defined ones used via the global-accessor)
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam T The type of the storage used to store data
     */
    template < uint_t I, typename Storage, typename Condition = bool >
    struct arg {
        typedef Storage storage_type;
        typedef typename Storage::iterator_type iterator_type;
        typedef typename Storage::value_type value_type;
        typedef static_uint< I > index_type;
        typedef static_uint< I > index;

        template < typename Storage2 >
        arg_storage_pair< arg< I, storage_type >, Storage2 > operator=(Storage2 &ref) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< Storage2, storage_type >::value),
                "there is a mismatch between the storage types used by the arg placeholders and the storages really "
                "instantiated. Check that the placeholders you used when constructing the aggregator_type are in the "
                "correctly assigned and that their type match the instantiated storages ones");

            return arg_storage_pair< arg< I, storage_type >, Storage2 >(&ref);
        }

        static void info(std::ostream &out_s) {
#ifdef VERBOSE
            out_s << "Arg on real storage with index " << I;
#endif
        }
    };

    /**
     * This specialization is made for the standard storages (not user-defined)
     * which have to contain a storage_info type, and can define a location_type
     */
    template < uint_t I, typename Storage >
    struct arg< I, Storage, typename boost::enable_if< typename is_any_storage< Storage >::type, bool >::type > {
        typedef Storage storage_type;
        typedef typename Storage::iterator_type iterator_type;
        typedef typename Storage::value_type value_type;
        typedef static_uint< I > index_type;
        typedef static_uint< I > index;

// location type is only used by other grids, supported only for cxx11
#ifdef CXX11_ENABLED
        using location_type = typename Storage::storage_info_type::index_type;
#endif

        template < typename Storage2 >
        arg_storage_pair< arg< I, storage_type >, Storage2 > operator=(Storage2 &ref) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< Storage2, storage_type >::value),
                "there is a mismatch between the storage types used by the arg placeholders and the storages really "
                "instantiated. Check that the placeholders you used when constructing the aggregator_type are in the "
                "correctly assigned and that their type match the instantiated storages ones");

            return arg_storage_pair< arg< I, storage_type >, Storage2 >(&ref);
        }

        static void info(std::ostream &out_s) {
#ifdef VERBOSE
            out_s << "Arg on real storage with index " << I;
#endif
        }
    };

    template < typename T >
    struct is_arg : boost::mpl::false_ {};

    template < uint_t I, typename Storage >
    struct is_arg< arg< I, Storage > > : boost::mpl::true_ {};

    template < typename T >
    struct arg_index;

    /** true in case of non temporary storage arg*/
    template < uint_t I, typename Storage >
    struct arg_index< arg< I, Storage > > : boost::mpl::integral_c< int, I > {};

    template < typename T >
    struct is_storage_arg : boost::mpl::false_ {};

    template < uint_t I, typename Storage >
    struct is_storage_arg< arg< I, Storage > > : is_storage< Storage > {};

    /**
     * @struct arg_hods_data_field
     * metafunction that determines if an arg type is holding the storage type of a data field
     */
    template < typename Arg >
    struct arg_holds_data_field;

    template < uint_t I, typename Storage >
    struct arg_holds_data_field< arg< I, Storage > > {
        typedef typename storage_holds_data_field< Storage >::type type;
    };

} // namespace gridtools
