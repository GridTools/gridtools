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
#include "storage_list.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/reversed_range.hpp"
#include "data_field_metafunctions.hpp"

namespace gridtools {

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
                GRIDTOOLS_STATIC_ASSERT(is_data_field< Storage >::value,
                    "\"swap\" can only be called with instances of type \"data_field\" ");
                typename Storage::pointer_type tmp = storage_.template get< SnapshotFrom, DimFrom >();
                tmp.set_externally_managed(false);
                storage_.template get< SnapshotFrom, DimFrom >() = storage_.template get< SnapshotTo, DimTo >();
                storage_.template get< SnapshotTo, DimTo >() = tmp;
            }
        };
    };

    /**@brief shifts the snapshots in one data field dimension

       @tparam Dim the data field dimension

       it cycles, i.e. the pointer to the last snapshots becomes the first
       (so that the storage is overwritten)

       syntax:
       advance<2>()(storage_);

       \tparam Dim the component to cycle
    */
    template < ushort_t Dim >
    struct cycle {

        template < typename Storage >
        struct shift {
          private:
            Storage &m_storage;

          public:
            shift(Storage &storage_) : m_storage(storage_) {}

            template < typename Id >
            void operator()(Id) {
                swap< Id::value - 1, Dim >::template with< Id::value, Dim >::apply(m_storage);
            }
        };

        template < typename Storage >
        static void apply(Storage &storage_) {
            GRIDTOOLS_STATIC_ASSERT(is_data_field< typename Storage::super >::value,
                "\"advance\" can only be called with instanced of type \"data_field\" ");

            boost::mpl::for_each<
                boost::mpl::range_c< ushort_t, 1, impl_::width_t< typename Storage::super, Dim >::value > >(
                shift< Storage >(storage_));
        }
    };

    struct cycle_all {

        template < typename Storage >
        struct call_apply {
          private:
            Storage &m_storage;

          public:
            call_apply(Storage &storage_) : m_storage(storage_) {}

            template < typename Id >
            void operator()(Id) {
                cycle< Id::value >::apply(m_storage);
            }
        };

        template < typename Storage >
        static void apply(Storage &storage_) {
            GRIDTOOLS_STATIC_ASSERT(is_data_field< typename Storage::super >::value,
                "\"advance\" can only be called with instanced of type \"data_field\" ");
            boost::mpl::for_each<
                typename boost::mpl::range_c< ushort_t, 0, Storage::super::traits::n_dimensions >::type >(
                call_apply< Storage >(storage_));
        }
    };

    /**@brief implements the field structure

       It is a collection of arbitrary length \ref gridtools::storage_list "storage lists".

       \include storage.dox

    */
    template < typename First, typename... StorageExtended >
    struct data_field : public dimension_extension_traits< First, StorageExtended... >::type {
        template < typename PT, typename MD, ushort_t FD >
        using type_tt = data_field< typename First::template type_tt< PT, MD, FD >,
            typename StorageExtended::template type_tt< PT, MD, FD >... >;

        typedef data_field< First, StorageExtended... > type;
        typedef typename dimension_extension_traits< First, StorageExtended... >::type super;
        typedef dimension_extension_traits< First, StorageExtended... > traits;
        typedef typename super::pointer_type pointer_type;
        typedef typename super::value_type value_type;
        typedef typename super::basic_type basic_type;
        static const short_t n_width = sizeof...(StorageExtended) + 1;

        /**@brief default constructor*/
        template < typename... ExtraArgs >
        data_field(pointer< typename basic_type::storage_info_type const > meta_data_, ExtraArgs const &... args_)
            : super(meta_data_, args_...) {}

        // /**@brief device copy constructor*/
        // template < typename T >
        // __device__ data_field(T const &other)
        //     : super(other) {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field() {}

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           @param field the pointer to the input data field
           @tparam dimension specifies which field dimension we want to access
        */
        template < uint_t dimension = 1 >
        GT_FUNCTION void push_front(pointer_type &field) { // copy constructor

            // cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when
            // accessing the storage (stateless storage)

            /*If the following assertion fails your field dimension is smaller than the dimension you are trying to
             * access*/
            BOOST_STATIC_ASSERT(n_width > dimension);
            /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each
             * dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width <= traits::n_fields);
            uint_t const indexFrom = _impl::access< n_width - dimension, traits >::type::n_fields;
            uint_t const indexTo = _impl::access< n_width - dimension - 1, traits >::type::n_fields;
            super::push_front(field, indexFrom, indexTo);
        }

        /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template < uint_t dimension = 1 >
        GT_FUNCTION void push_front(pointer_type &field, typename super::value_type const &value) { // copy constructor
            for (uint_t i = 0; i < this->m_meta_data->size(); ++i)
                field[i] = value;
            push_front< dimension >(field);
        }

        /**@brief sets the given storage as the nth snapshot of a specific field dimension

           @tparam field_dim the given field dimenisons
           @tparam snapshot the snapshot of dimension field_dim to be set
           @param field the input storage
        */
        template < short_t snapshot = 0, short_t field_dim = 0 >
        void set(pointer_type &field) {

            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to set a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to set a field dimension out of bound");
            super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot] = field;
        }

        /**@brief sets the given storage as the nth snapshot of a specific field dimension and initialize the storage
           with an input constant value

           @tparam field_dim the given field dimenisons
           @tparam snapshot the snapshot of dimension field_dim to be set
           @param field the input storage
           @param val the initializer value
        */
        template < short_t snapshot = 0, short_t field_dim = 0 >
        void set(/* pointer_type& field,*/ typename super::value_type const &val) {

            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to set a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to set a field dimension out of bound");
            for (uint_t i = 0; i < this->m_meta_data->size(); ++i)
                (super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot])[i] = val;
        }

        /**@brief sets the given storage as the nth snapshot of a specific field dimension and initialize the storage
           with an input lambda function
           TODO: this should be merged with the boundary conditions code (repetition)

           @tparam field_dim the given field dimenisons
           @tparam snapshot the snapshot of dimension field_dim to be set
           @param field the input storage
           @param lambda the initializer function
        */
        template < short_t snapshot = 0, short_t field_dim = 0 >
        void set(typename super::value_type (*lambda)(uint_t const &, uint_t const &, uint_t const &)) {

            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to set a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to set a fielddimension out of bound");
            for (uint_t i = 0; i < this->m_meta_data->template dim< 0 >(); ++i)
                for (uint_t j = 0; j < this->m_meta_data->template dim< 1 >(); ++j)
                    for (uint_t k = 0; k < this->m_meta_data->template dim< 2 >(); ++k)
                        (super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields +
                                         snapshot])[this->m_meta_data->index(i, j, k)] = lambda(i, j, k);
        }

        /**@brief gets the given storage as the nth snapshot of a specific field dimension

           @tparam field_dim the given field dimenisons
           @tparam snapshot the snapshot of dimension field_dim to be set
        */
        template < short_t snapshot = 0, short_t field_dim = 0 >
        pointer_type &get() {
            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to get a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to get a field dimension out of bound");
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(snapshot < super::super::field_dimensions, "nasty error");
            GRIDTOOLS_STATIC_ASSERT((_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot <
                                        super::super::field_dimensions),
                "nasty error");
#endif
            return super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot];
        }

        template < short_t snapshot = 0, short_t field_dim = 0 >
        pointer_type const &get() const {
            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to get a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to get a field dimension out of bound");
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(snapshot < super::super::field_dimensions, "nasty error");
            GRIDTOOLS_STATIC_ASSERT((_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot <
                                        super::super::field_dimensions),
                "nasty error");
#endif
            return super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields + snapshot];
        }

        /**@brief gets a given value at the given field i,j,k coordinates

           @tparam field_dim the given field dimenisons
           @tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           @param i index in the horizontal direction
           @param j index in the horizontal direction
           @param k index in the vertical direction
        */
        template < short_t snapshot = 0, short_t field_dim = 0, typename... Int >
        typename super::value_type &get_value(Int... args) {

            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to get a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to get a field dimension out of bound");
            return super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields +
                                   snapshot][this->m_meta_data->index(args...)];
        }

        /**@brief gets a given value at the given field i,j,k coordinates

           same as the previous one, but returning a constant reference
        */
        template < short_t snapshot = 0, short_t field_dim = 0, typename... Int >
        typename super::value_type const &get_value(Int... args) const {

            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to get a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to get a field_dimension out of bound");
            return super::m_fields[_impl::access< n_width - (field_dim), traits >::type::n_fields +
                                   snapshot][this->m_meta_data->index(args...)];
        }
    };

} // namespace gridtools
#endif
