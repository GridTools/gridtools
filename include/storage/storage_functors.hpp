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
#pragma once
#include "base_storage.hpp"
#include <boost/fusion/include/is_sequence.hpp>

namespace gridtools {

    /**
        @class
        @brief functor to copy a (smart) storage pointer to a destination only if the destination
        is not temporary
        @tparam Orig origin type
    */
    template < typename Orig >
    struct copy_pointers_set_functor {

      private:
        Orig const &m_orig;

      public:
        copy_pointers_set_functor(Orig const &orig_) : m_orig(orig_) {}

        /**
           @brief operator copying an element of a set into a destination only if the destination type
           is not a pointer to a temporary storage
         */
        template < typename Dest >
        GT_FUNCTION_WARNING void operator()(Dest &dest_) const {
            GRIDTOOLS_STATIC_ASSERT(boost::fusion::traits::is_sequence< Orig >::type::value, " The origin\
 container is not a fusion sequence.");

            if (!Dest::value_type::is_temporary)
                dest_ = boost::fusion::at_key< Dest >(m_orig);
        }
    };

    /**
       @struct
       @brief copies (smart) storage pointers from one container to another only if the destination
       is not temporary
       @tparam DestCont destination container
       @tparam OrigCont origin container
    */
    template < typename DestCont, typename OrigCont >
    struct copy_pointers_functor {

        GRIDTOOLS_STATIC_ASSERT(boost::fusion::traits::is_sequence< DestCont >::type::value, " The destination\
 container is not a fusion sequence.");
        GRIDTOOLS_STATIC_ASSERT(boost::fusion::traits::is_sequence< OrigCont >::type::value, " The origin\
 container is not a fusion sequence.");

        /**
           @brief constructor

           @param dc destination sequence
           @param oc origin sequence
        */
        copy_pointers_functor(DestCont &dc, OrigCont const &oc) : m_dc(dc), m_oc(oc) {}

        template < typename Index >
        GT_FUNCTION_WARNING void operator()(const Index &) const {
            GRIDTOOLS_STATIC_ASSERT(is_static_integral< Index >::type::value, "wrong type");
            assign< Index >();
        }

      private:
        //! do not copy pointers in case storage is a temporary
        template < typename Index >
        GT_FUNCTION_WARNING void assign(
            typename boost::enable_if_c< is_temporary_storage< typename boost::remove_pointer<
                typename boost::mpl::at< DestCont, Index >::type >::type >::value >::type * = 0) const {
            GRIDTOOLS_STATIC_ASSERT(is_static_integral< Index >::type::value, "wrong type");
        }

        /**
           @brief assigning the pointers (with the = operator)
        */
        template < typename Index >
        GT_FUNCTION_WARNING void assign(
            typename boost::disable_if_c< is_temporary_storage< typename boost::remove_pointer<
                typename boost::mpl::at< DestCont, Index >::type >::type >::value >::type * = 0) const {
            GRIDTOOLS_STATIC_ASSERT(is_static_integral< Index >::type::value, "wrong type");
            boost::fusion::at< Index >(m_dc) = boost::fusion::at< Index >(m_oc);
        }

        DestCont &m_dc;
        OrigCont const &m_oc;
    };

    /**@brief Functor updating the pointers on the device */
    struct update_pointer {

        template < typename StorageType >
        GT_FUNCTION_WARNING void operator()(pointer< StorageType > &s) const {

#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(is_storage< StorageType >::value, "wrong type");
#endif

            if (s.get()) {
                // update pointer should not copy the data!!
                copy_data_impl< StorageType >(s);
                s->clone_to_device();
            }
        }

        /**
           @brief Functor updating the pointers on the device

           separate (for the moment) overload for the metadata
           since the metadata is const
        */
        template < typename MetaDataType >
        GT_FUNCTION_WARNING void operator()(pointer< const MetaDataType > &s) const {

            GRIDTOOLS_STATIC_ASSERT((is_meta_storage< MetaDataType >::value), "type error");
            if (s.get()) {
                s->clone_to_device();
            }
        }

      private:
        // we do not copy data into the gpu in case of a generic accessor
        template < typename StorageType >
        GT_FUNCTION_WARNING void copy_data_impl(pointer< StorageType > &s,
            typename boost::disable_if_c< is_storage< StorageType >::value >::type * = 0) const {}

        // we do not copy data into the gpu in case of a temporary
        template < typename StorageType >
        GT_FUNCTION_WARNING void copy_data_impl(pointer< StorageType > &s,
            typename boost::enable_if_c< is_temporary_storage< StorageType >::value >::type * = 0,
            typename boost::enable_if_c< is_storage< StorageType >::value >::type * = 0) const {}

        template < typename StorageType >
        GT_FUNCTION_WARNING void copy_data_impl(pointer< StorageType > &s,
            typename boost::disable_if_c< is_temporary_storage< StorageType >::value >::type * = 0,
            typename boost::enable_if_c< is_storage< StorageType >::value >::type * = 0) const {
            s->h2d_update();
        }
    };

    struct call_d2h {
        template < typename StorageType >
        GT_FUNCTION void operator()(pointer< StorageType > arg) const {
#ifndef __CUDA_ARCH__
            do_impl< StorageType >(arg, static_cast< typename is_no_storage_type_yet< StorageType >::type * >(0));
#endif
        }

      private:
        template < typename StorageType >
        GT_FUNCTION void do_impl(pointer< StorageType > arg,
            typename boost::enable_if_c< is_no_storage_type_yet< StorageType >::value >::type * = 0) const {}
        template < typename StorageType >
        GT_FUNCTION void do_impl(pointer< StorageType > arg,
            typename boost::enable_if_c< is_storage< StorageType >::value >::type * = 0,
            typename boost::disable_if_c< is_no_storage_type_yet< StorageType >::value >::type * = 0) const {
            arg->d2h_update();
        }

        template < typename StorageType >
        GT_FUNCTION void do_impl(pointer< StorageType > arg,
            typename boost::disable_if_c< is_storage< StorageType >::value >::type * = 0,
            typename boost::disable_if_c< is_no_storage_type_yet< StorageType >::value >::type * = 0) const {}
    };

} // namespace gridtools
