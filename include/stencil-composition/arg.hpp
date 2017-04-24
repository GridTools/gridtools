/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used
   in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.
*/

#pragma once

#include <iosfwd>

#include "../common/defs.hpp"
#include "../common/pointer.hpp"
#include "arg_fwd.hpp"
#include "arg_metafunctions.hpp"
#include "arg_metafunctions_fwd.hpp"
#include "storage-facility.hpp"

namespace gridtools {

    template < typename T >
    struct is_arg : boost::mpl::false_ {};

    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct is_arg< arg< I, Storage, Location, Temporary > > : boost::mpl::true_ {};

    template < typename T >
    struct is_tmp_arg;

    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct is_tmp_arg< arg< I, Storage, Location, Temporary > > : boost::mpl::bool_< Temporary > {};

    /** @brief binding between the placeholder (\tparam ArgType) and the storage (\tparam Storage)*/
    template < typename ArgType, typename Storage >
    struct arg_storage_pair {

        GRIDTOOLS_STATIC_ASSERT(is_arg< ArgType >::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename ArgType::storage_t, Storage >::type::value),
            "Storage type not compatible with placeholder storage type, when associating placeholder to actual "
            "storage");

      public:
        pointer< Storage > ptr;

        arg_storage_pair() {}
        arg_storage_pair(Storage *storage) : ptr(pointer< Storage >(storage)) {}
        arg_storage_pair(arg_storage_pair const &other) : ptr(other.ptr) {}

        typedef ArgType arg_t;
        typedef Storage storage_t;
    };

    template < typename T >
    struct is_arg_storage_pair : boost::mpl::false_ {};

    template < typename ArgType, typename Storage >
    struct is_arg_storage_pair< arg_storage_pair< ArgType, Storage > > : boost::mpl::true_ {};

    template < typename T >
    struct is_arg_storage_pair_to_tmp : boost::mpl::false_ {};

    template < typename ArgType, typename Storage >
    struct is_arg_storage_pair_to_tmp< arg_storage_pair< ArgType, Storage > >
        : boost::mpl::bool_< ArgType::is_temporary > {};

    /**
     * Type to create placeholders for data fields.
     *
     * There is a specialization for the case in which T is a temporary.
     * The default version applies to all the storage classes (including
     * user-defined ones used via the global-accessor)
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam Storage The type of the storage used to store data
     * @tparam LocationType the location type of the storage of the placeholder
     * @tparam is_temporary_storage determines whether the placeholder holds a temporary or normal storage
     */
    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct arg {
        typedef Storage storage_t;
        typedef static_uint< I > index_t;
        typedef Location location_t;
        constexpr static bool is_temporary = Temporary;

        template < typename Storage2 >
        arg_storage_pair< arg< I, storage_t, Location, Temporary >, Storage2 > operator=(Storage2 &ref) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< Storage2, storage_t >::value),
                "there is a mismatch between the storage types used by the arg placeholders and the storages really "
                "instantiated. Check that the placeholders you used when constructing the aggregator_type are in the "
                "correctly assigned and that their type match the instantiated storages ones");

            return arg_storage_pair< arg< I, storage_t, Location, Temporary >, Storage2 >(&ref);
        }

        static void info(std::ostream &out_s) {
#ifdef VERBOSE
            out_s << "Arg on real storage with index " << I;
#endif
        }
    };

    template < typename T >
    struct arg_index;

    /** true in case of non temporary storage arg*/
    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct arg_index< arg< I, Storage, Location, Temporary > > : boost::mpl::integral_c< int, I > {};

    template < typename T >
    struct is_storage_arg : boost::mpl::false_ {};

    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct is_storage_arg< arg< I, Storage, Location, Temporary > > : is_storage< Storage > {};

    /**
     * @struct arg_hods_data_field
     * metafunction that determines if an arg type is holding the storage type of a data field
     */
    template < typename Arg >
    struct arg_holds_data_field;

    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct arg_holds_data_field< arg< I, Storage, Location, Temporary > > : is_data_store_field< Storage > {};

} // namespace gridtools
