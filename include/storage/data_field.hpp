#pragma once
#include "storage_list.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/reversed_range.hpp"

namespace gridtools {
    /** @brief traits class defining some useful compile-time counters
     */
    template < typename First, typename... StorageExtended >
    struct dimension_extension_traits {
        // total number of snapshots in the discretized data field
        static const ushort_t n_fields = First::n_width + dimension_extension_traits< StorageExtended... >::n_fields;
        // the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width = First::n_width;
        // the number of dimensions (i.e. the number of different storage_lists)
        static const ushort_t n_dimensions = dimension_extension_traits< StorageExtended... >::n_dimensions + 1;
        // the current field extension
        // n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
        typedef storage_list< First, n_fields - 1 > type;
        // typedef First type;
        typedef dimension_extension_traits< StorageExtended... > super;
    };

    /**@brief fallback in case the snapshot we try to access exceeds the width dimension assigned to a discrete scalar
     * field*/
    struct dimension_extension_null {
        static const ushort_t n_fields = 0;
        static const short_t n_width = 0;
        static const ushort_t n_dimensions = 0;
        typedef struct error_index_too_large1 {
        } type;
        typedef struct error_index_too_large2 { } super; };

    /**@brief template specialization at the end of the recustion.*/
    template < typename First >
    struct dimension_extension_traits< First > {
        static const ushort_t n_fields = First::n_width;
        static const short_t n_width = First::n_width;
        static const ushort_t n_dimensions = 1;
        typedef First type;
        typedef dimension_extension_null super;
    };

    template < typename T >
    struct get_fields {
        using type = static_int< T::n_fields >;
    };

    template < typename T >
    struct get_value_ {
        using type = static_int< T::value >;
    };

    template < typename T >
    struct get_width {
        using type = static_int< T::n_width >;
    };

    /** @brief metafunction to compute the number of total snapshots present in the data field
        (sum of storage_list::n_width) before
        the ID-th storage list*/
    template < typename Storage, uint_t Id, uint_t IdMax >
    struct compute_storage_offset {

        GRIDTOOLS_STATIC_ASSERT(IdMax >= Id && Id >= 0, "Library internal error");
        typedef typename boost::mpl::eval_if_c< IdMax - Id == 0,
            get_fields< typename Storage::super >,
            get_value_< compute_storage_offset< typename Storage::super, Id + 1, IdMax > > >::type type;
        static const uint_t value = type::value;
    };

    /** @brief metafunction to compute the number of snapshots present in the ID-th storage_list
        (storage_list::n_width)
    */
    template < typename Storage, uint_t Id, uint_t IdMax >
    struct compute_storage_list_width {

        GRIDTOOLS_STATIC_ASSERT(IdMax >= Id && Id >= 0, "Library internal error");
        typedef typename boost::mpl::eval_if_c< IdMax - Id == 0,
            get_width< Storage >,
            get_width< compute_storage_list_width< typename Storage::super, Id + 1, IdMax > > >::type type;
        static const uint_t value = type::value;
    };

    template < typename T >
    struct is_data_field : public boost::mpl::false_ {};

    namespace impl_ {
        /**@brief syntactic sugar*/
        template < typename Storage, uint_t Id >
        using offset_t = compute_storage_offset< typename Storage::traits, Id, Storage::traits::n_dimensions - 1 >;

        /**@brief syntactic sugar*/
        template < typename Storage, uint_t Id >
        using width_t = compute_storage_list_width< typename Storage::traits, Id, Storage::traits::n_dimensions - 1 >;
    }

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
    */
    template < ushort_t Dim >
    struct advance {

        template < typename Storage >
        struct shift {
          private:
            Storage &m_storage;

          public:
            shift(Storage &storage_) : m_storage(storage_) {}

            template < typename Id >
            void operator()(Id) {
                std::cout << "ID: " << Id::value << std::endl;
                m_storage.fields_view()[Id::value] = m_storage.fields_view()[Id::value - 1];
            }
        };

        template < typename Storage >
        static void apply(Storage &storage_) {
            GRIDTOOLS_STATIC_ASSERT(is_data_field< Storage >::value,
                "\"advance\" can only be called with instanced of type \"data_field\" ");
            // save last snapshot
            typename Storage::pointer_type tmp =
                storage_
                    .fields_view()[impl_::width_t< Storage, Dim >::value + impl_::offset_t< Storage, Dim >::value - 1];

            typedef typename reversed_range< ushort_t,
                1 + impl_::offset_t< Storage, Dim >::value,
                impl_::width_t< Storage, Dim >::value + impl_::offset_t< Storage, Dim >::value >::type range_t;

            boost::mpl::for_each< range_t >(shift< Storage >(storage_));

            // restore the first snapshot
            storage_.fields_view()[impl_::offset_t< Storage, Dim >::value] = tmp;
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
        typedef typename super::basic_type basic_type;
        static const short_t n_width = sizeof...(StorageExtended) + 1;

        /**@brief default constructor*/
        template < typename... ExtraArgs >
        data_field(typename basic_type::storage_info_type const *meta_data_, ExtraArgs const &... args_)
            : super(meta_data_, args_...) {}

        /**@brief device copy constructor*/
        template < typename T >
        __device__ data_field(T const &other)
            : super(other) {}

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

        /**@biref sets the given storage as the nth snapshot of a specific field dimension

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

        /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage
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

        /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage
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

        /**@biref gets the given storage as the nth snapshot of a specific field dimension

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

        /**@biref gets a given value at the given field i,j,k coordinates

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
            return get< snapshot, field_dim >()[this->m_meta_data->index(args...)];
        }

        /**@biref gets a given value at the given field i,j,k coordinates

           same as the previous one, but returning a constant reference
        */
        template < short_t snapshot = 0, short_t field_dim = 0, typename... Int >
        typename super::value_type const &get_value(Int... args) const {
            GRIDTOOLS_STATIC_ASSERT((snapshot < _impl::access< n_width - (field_dim)-1, traits >::type::n_width),
                "trying to get a snapshot out of bound");
            GRIDTOOLS_STATIC_ASSERT((field_dim < traits::n_dimensions), "trying to get a field_dimension out of bound");
            return get< snapshot, field_dim >()[this->m_meta_data->index(args...)];
        }

        /**@biref ODE advancing for a single dimension

           it advances the supposed finite difference scheme of one step for a specific field dimension
           @tparam dimension the dimension to be advanced
           @param offset the number of steps to advance
        */
        template < uint_t dimension = 1 >
        GT_FUNCTION void advance() {
            BOOST_STATIC_ASSERT(dimension < traits::n_dimensions);
            uint_t const indexFrom = _impl::access< dimension, traits >::type::n_fields;
            uint_t const indexTo = _impl::access< dimension - 1, traits >::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

        /**@biref ODE advancing for all dimension

           shifts the rings of solutions of one position,
           it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all() { _impl::advance_recursive< n_width >::apply(const_cast< data_field * >(this)); }
    };

    template < typename First, typename... StorageExtended >
    struct is_data_field< data_field< First, StorageExtended... > > : public boost::mpl::true_ {};

    template < typename T >
    struct storage;

    template < typename First, typename... StorageExtended >
    struct is_data_field< storage< data_field< First, StorageExtended... > > > : public boost::mpl::true_ {};

    template < typename F, typename... T >
    std::ostream &operator<<(std::ostream &s, data_field< F, T... > const &) {
        return s << "field storage";
    }

} // namespace gridtools
#endif
