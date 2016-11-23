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
#include "base_storage.hpp"
#ifdef CXX11_ENABLED
namespace gridtools {
    /** @brief storage class containing a buffer of data snapshots

        it is a list of \ref gridtools::base_storage "storages"

        \include storage.dox

    */
    template < typename Storage, short_t ExtraWidth >
    struct storage_list : public Storage {

        typedef storage_list traits;
        static const uint_t n_dimensions = 1;
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
        typedef typename super::iterator iterator;
        typedef typename super::value_type value_type;

        /**@brief constructor*/
        template < typename... Args >
        storage_list(typename basic_type::storage_info_type const *meta_data_, Args const &... args_)
            : super(meta_data_, args_...) {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list() {}

        /**@brief device copy constructor*/
        template < typename T >
        __device__ storage_list(T const &other) : super(other) {
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
        storage_list(typename basic_type::storage_info_type const *meta_data_, Args const &... args_)
            : super(meta_data_, args_...) {}

        // default constructor
        storage_list(typename basic_type::storage_info_type const *meta_data_) : super(meta_data_) {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list() {}

        /**dimension number of snaphsots for the current field dimension*/
        static const ushort_t n_width = Storage::n_width;

        /**@brief device copy constructor*/
        template < typename T >
        __device__ storage_list(T const &other) : super(other) {}
    };

    template < typename T >
    struct is_storage_list : boost::mpl::false_ {};

    template < typename Storage, short_t ExtraWidth >
    struct is_storage_list< storage_list< Storage, ExtraWidth > > : boost::mpl::true_ {};

} // namespace gridtools
#endif
