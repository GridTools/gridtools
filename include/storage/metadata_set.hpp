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
#include <boost/fusion/include/as_set.hpp>
#include "common/generic_metafunctions/is_sequence_of.hpp"

/**
@file
@brief implementing a set
*/

namespace gridtools {

    template < typename T >
    struct is_any_storage;

    /**
       @brief class that given a generic MPL sequence creates a fusion set.

       The interface of this class allows to insert and get elements of the sequence give its type.
       It also allows to query if the element corresponding to a given type has been or not initialized

       It is used to hold the list of meta-storages, in which a 1-1 relation is needed between instances
       and types.
     */
    template < typename Sequence >
    struct metadata_set {
        GRIDTOOLS_STATIC_ASSERT(boost::mpl::is_sequence< Sequence >::value, "internal error: not a sequence");
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< Sequence, is_pointer >::value), "internal error: not a sequence of pointers");
        typedef typename boost::fusion::result_of::as_set< Sequence >::type set_t;

      private:
        set_t m_set;

      public:
        GT_FUNCTION
        metadata_set() : m_set(){};

        /**
           @brief device copy constructor
         */
        __device__ metadata_set(metadata_set const &other) : m_set(other.m_set){};

        /**
           @brief inserts a new instance in the sequence
           NOTE: pass by value
        */
        template < typename T >
        GT_FUNCTION void insert(T new_instance) {
            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key< set_t, T >::type::value),
                "the type used for the lookup in the metadata set is not present in the set. Did you use the correct "
                "type as meta storage?");
            boost::fusion::at_key< T >(m_set) = new_instance;
        }

        /**
           @brief returns the raw pointer given a key

           \tparam T key
        */
        template < typename T >
        GT_FUNCTION T const &get() {
            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key< set_t, T >::type::value),
                "Internal error: calling metadata_set::get with a metadata type which has not been defined.");
            assert(present< T >()); // must be initialized
            return boost::fusion::at_key< T >(m_set);
        }

        /**@brief returns the sequence by non-const reference*/
        GT_FUNCTION
        set_t &sequence_view() { return m_set; }

        /**@brief returns the sequence by const reference*/
        GT_FUNCTION
        set_t const &sequence_view() const { return m_set; }

        /**@bief queries if the given key corresponds to a pointer which has been initialized*/
        template < typename T >
        GT_FUNCTION bool present() {
            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key< set_t, T >::type::value),
                "internal error: calling metadata_set::present with a metadata type which has not been defined");
            return boost::fusion::at_key< T >(m_set).get();
        }
    };

    template < typename T >
    struct is_metadata_set : boost::mpl::false_ {};

    template < typename T >
    struct is_metadata_set< metadata_set< T > > : boost::mpl::true_ {};

    template < typename U >
    struct is_storage;

    /** inserts an element in the set if it is not present

        used for the metadata_set in the aggregator_type
    */
    template < typename Sequence, typename Arg >
    struct insert_if_not_present {

#ifdef PEDANTIC // disabling in case of generic accessors
        GRIDTOOLS_STATIC_ASSERT(is_any_storage< Arg >::value,
            "if you are using generic accessors disable the pedantic mode. Otherwise most probably you used in the "
            "aggregator_type constructor a storage type which is not supported.");
#endif
        GRIDTOOLS_STATIC_ASSERT(is_metadata_set< Sequence >::type::value, "wrong type");

      private:
        Sequence &m_seq;
        typedef typename Arg::storage_ptr_t arg_ptr_t;
        pointer< const arg_ptr_t > m_arg_ptr;

      public:
        GT_FUNCTION
        insert_if_not_present(Sequence &seq_, Arg const &arg_) : m_seq(seq_), m_arg_ptr(arg_.get_storage_pointer()) {}

        GT_FUNCTION
        void operator()() const {
            if (!m_seq.template present< pointer< const typename Arg::storage_info_type > >())
                m_seq.insert((**m_arg_ptr).meta_data_ptr());
            else
                assert(
                    *m_seq.template get< pointer< const typename Arg::storage_info_type > >() ==
                        *(**m_arg_ptr).meta_data_ptr() &&
                    "the passed storages contain different meta data (e.g., different dimension) which is not valid.");
        }
    };
}
