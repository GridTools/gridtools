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

#include <iosfwd>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/mpl/max_element.hpp>
#include "base_storage_impl.hpp"
#include "../common/array.hpp"
#include "../common/array_addons.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"
#include "../common/explode_array.hpp"
#include "../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/generic_metafunctions/variadic_assert.hpp"
#include "../common/offset_metafunctions.hpp"

/**
   @file
   @brief basic file containing the storage meta information container
 */

/**
    @class
    @brief class containing the meta storage information
*/
namespace gridtools {

    /**
     * @brief Type to indicate that the type is not decided yet
     */
    template < typename RegularMetaStorageType >
    struct no_meta_storage_type_yet {
        typedef RegularMetaStorageType type;
        typedef typename type::index_type index_type;
        typedef typename type::layout layout;
        typedef typename type::alignment_t alignment_t;
        typedef typename type::halo_t halo_t;
        static const ushort_t space_dimensions = type::space_dimensions;
        static const bool is_temporary = type::is_temporary;
    };

    template < typename T >
    struct is_meta_storage : boost::mpl::false_ {};

    /**fwd declaration*/
    template < typename Index,
        typename Layout,
        bool IsTemporary
#ifdef CXX11_ENABLED
        ,
        typename... Tiles
#else
        ,
        typename TileI = int,
        typename TileJ = int
#endif
        >
    struct meta_storage_base;

    /**@brief class containing the storage meta information

       \tparam Index an index used to differentiate the types also when there's only runtime
       differences (e.g. only the storage dimensions differ)
       \tparam Layout the map of the layout in memory
       \tparam IsTemporary boolean flag set to true when the storage is a temporary one
     */
    template < typename Index, typename Layout, bool IsTemporary >
    struct meta_storage_base< Index,
        Layout,
        IsTemporary
#ifndef CXX11_ENABLED
        ,
        int,
        int
#endif
        > {
        typedef meta_storage_base< Index,
            Layout,
            IsTemporary
#ifndef CXX11_ENABLED
            ,
            int,
            int
#endif
            > type;
        typedef Layout layout;
        typedef Index index_type;

        static const bool is_temporary = IsTemporary;
        static const ushort_t n_width = 1;
        static const ushort_t space_dimensions = layout::length;
        typedef meta_storage_base< Index, Layout, IsTemporary > basic_type;

      protected:
        array< uint_t, space_dimensions > m_dims;
        // control your instincts: changing the following
        // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
        array< int_t, space_dimensions > m_strides;

      public:
#ifdef CXX11_ENABLED
        template < typename T, typename U, bool B, typename... D >
        friend std::ostream &operator<<(std::ostream &, meta_storage_base< T, U, B, D... > const &);
#else
        template < typename T, typename U, bool B, typename T1, typename T2 >
        friend std::ostream &operator<<(std::ostream &, meta_storage_base< T, U, B, T1, T2 > const &);
#endif

        /**
           @brief empty constructor
        */
        GT_FUNCTION
        constexpr meta_storage_base() {}

#ifdef CXX11_ENABLED
#ifndef __CUDACC__
        template < class... IntTypes, typename Dummy = all_integers< IntTypes... > >
        GT_FUNCTION void setup(IntTypes const &... dims_) {
            m_dims = array< uint_t, space_dimensions >(dims_...);
            m_strides = array< int_t, space_dimensions >(
                _impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply(dims_...));
        }
#else
        template < class First,
            class... IntTypes,
            typename Dummy = typename boost::enable_if_c< boost::is_integral< First >::type::value,
                bool >::type // nvcc does not get it
            >
        GT_FUNCTION void setup(First first_, IntTypes const &... dims_) {
            m_dims = array< uint_t, space_dimensions >{first_, dims_...};
            m_strides = array< int_t, space_dimensions >(
                _impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply(first_, dims_...));
        }
#endif //__CUDACC__

        GT_FUNCTION
        constexpr meta_storage_base(array< uint_t, space_dimensions > const &a)
            : m_dims(a), m_strides(explode< array< int_t, (short_t)(space_dimensions) >,
                             _impl::assign_all_strides< (short_t)(space_dimensions-1), layout > >(a)) {}

// variadic constexpr constructor

/**@brief generic multidimensional constructor given the space dimensions

   There are two possible types of storage dimension. One (space dimension) defines the number of indexes
   used to access a contiguous chunk of data. The other (field dimension) defines the number of pointers
   to the data chunks (i.e. the number of snapshots) contained in the storage. This constructor
   allows to create a storage with arbitrary space dimensions. The extra dimensions can be
   used e.g. to perform extra inner loops, besides the standard ones on i,j and k.

   The number of arguments must me equal to the space dimensions of the specific field (template parameter)
   NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
   at compile-time (e.g. in template metafunctions)
 */
#ifndef __CUDACC__
        template < class... IntTypes, typename Dummy = all_integers< IntTypes... > >
        GT_FUNCTION
            // we only use a constexpr in no debug mode, because we want to assert the sizes are uint in debug mode
            // constexpr does not allow code in the body
            constexpr meta_storage_base(IntTypes const &... dims_)
                : m_dims{(uint_t)dims_...},
                  m_strides(_impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply((uint_t)dims_...)) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) >= space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments smaller than its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) <= space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments larger than its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(boost::is_integral< IntTypes >::type::value...),
                "Error: Dimensions of metastorage must be specified as integer types. ");
        }
#else //__CUDACC__ nvcc does not get it: checks only the first argument
        template < class... IntTypes,
            typename Dummy = typename boost::enable_if_c<
                boost::is_integral<
                    typename boost::mpl::at_c< boost::mpl::vector< IntTypes... >, 0 >::type >::type::value,
                bool >::type >
        GT_FUNCTION constexpr meta_storage_base(IntTypes... dims_)
            : m_dims{(uint_t)dims_...},
              m_strides(_impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply(dims_...)) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) >= space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments smaller than its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) <= space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments larger than its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
        }

#endif //__CUDACC__

#ifndef __CUDACC__
        /**
           @brief constructor taking the space dimensions as static integers

           NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
           at compile-time (e.g. in template metafunctions)
         */
        template < typename... IntTypes, typename Dummy = all_static_integers< IntTypes... > >
        constexpr meta_storage_base(IntTypes... dims_)
            : m_dims{IntTypes::value...},
              m_strides(_impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply(IntTypes()...)) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) == space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments different from its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(boost::is_integral< IntTypes >::type::value...),
                "Error: Dimensions of metastorage must be specified as integer types. ");
        }

#else  //__CUDACC__

        template < typename First,
            typename... IntTypes,
            typename Dummy = typename boost::enable_if_c< is_static_integral< First >::type::value,
                bool >::type // nvcc does not get it
            >
        constexpr meta_storage_base(First first_, IntTypes... dims_)
            : m_dims{First::value, IntTypes::value...},
              m_strides(_impl::assign_all_strides< (short_t)(space_dimensions-1), layout >::apply(
                  First::value, IntTypes::value...)) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes) + 1 == space_dimensions, "you tried to initialize\
 a storage with a number of integer arguments different from its number of dimensions. \
 This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly\
 a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage\
 by defining the storage type using another layout_map.");
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_static_integral< IntTypes >::type::value...),
                "Error: Dimensions of metastorage must be specified as integer types. ");
        }
#endif //__CUDACC__
#else  // CXX11_ENABLED
        // TODO This is a bug, we should generate a constructor for array of dimensions space_dimensions
        GRIDTOOLS_STATIC_ASSERT(
            (space_dimensions == 3), "multidimensional storages are available only when C++11 is ON");
        GT_FUNCTION
        meta_storage_base(array< uint_t, 3 > const &a) : m_dims(a) {
            m_strides[0] = (((layout::template at_< 0 >::value < 0) ? 1 : m_dims[0]) *
                            ((layout::template at_< 1 >::value < 0) ? 1 : m_dims[1]) *
                            ((layout::template at_< 2 >::value < 0) ? 1 : m_dims[2]));
            m_strides[1] = ((m_strides[0] <= 1) ? 0 : layout::template find_val< 2, uint_t, 1 >(m_dims) *
                                                          layout::template find_val< 1, uint_t, 1 >(m_dims));
            m_strides[2] = ((m_strides[1] <= 1) ? 0 : layout::template find_val< 2, uint_t, 1 >(m_dims));
        }

        // non variadic non constexpr constructor
        GT_FUNCTION
        meta_storage_base(uint_t const &d1, uint_t const &d2, uint_t const &d3) : m_dims(d1, d2, d3) {
            m_strides[0] =
                (((layout::template at_< 0 >::value < 0) ? 1 : d1) * ((layout::template at_< 1 >::value < 0) ? 1 : d2) *
                    ((layout::template at_< 2 >::value < 0) ? 1 : d3));
            m_strides[1] = ((m_strides[0] <= 1) ? 0 : layout::template find_val< 2, short_t, 1 >(d1, d2, d3) *
                                                          layout::template find_val< 1, short_t, 1 >(d1, d2, d3));
            m_strides[2] = ((m_strides[1] <= 1) ? 0 : layout::template find_val< 2, short_t, 1 >(d1, d2, d3));
        }
#endif // CXX11_ENABLED

        /**
            @brief constexpr copy constructor

            copy constructor, used e.g. to generate the gpu clone of the storage metadata.
         */
        GT_FUNCTION
        constexpr meta_storage_base(meta_storage_base const &other)
            : m_dims(other.m_dims), m_strides(other.m_strides) {}

        /** @brief prints debugging information */
        void info(std::ostream &out_s) const { out_s << dim< 0 >() << "x" << dim< 1 >() << "x" << dim< 2 >() << " \n"; }

        /**@brief returns the size of the data field*/
        GT_FUNCTION
        constexpr uint_t size() const { // cast to uint_t
            return m_strides[0];
        }

        /** @brief returns the dimension fo the field along I*/
        GT_FUNCTION constexpr array< uint_t, space_dimensions > dims() const { return m_dims; }

        /** @brief returns the dimension fo the field along I*/
        template < ushort_t I >
        GT_FUNCTION constexpr uint_t dim() const {
            return m_dims[I];
        }

        /** @brief returns the dimension fo the field along I*/
        GT_FUNCTION
        constexpr uint_t dim(const ushort_t I) const { return m_dims[I]; }

        /**@brief returns the storage strides
         */
        GT_FUNCTION
        constexpr int_t const &strides(ushort_t i) const { return m_strides[i]; }

        /**@brief returns the storage strides
         */
        GT_FUNCTION
        constexpr int_t const *strides() const {
            GRIDTOOLS_STATIC_ASSERT(space_dimensions > 1, "less than 2D storage, is that what you want?");
            return (&m_strides[1]);
        }

#ifdef CXX11_ENABLED
        /**@brief straightforward interface*/
        template < typename... UInt, typename Dummy = all_integers< UInt... > >
        constexpr GT_FUNCTION int_t index(UInt const &... args_) const {
            return _index(strides(), args_...);
        }

        struct _impl_index {
            template < typename... UIntType >
            GT_FUNCTION static int_t apply(const type &me, UIntType... args) {
                return me.index(args...);
            }
        };

        template < typename... UInt, typename Dummy = all_static_integers< UInt... > >
        constexpr GT_FUNCTION uint_t index(uint_t const &first, UInt const &... args_) const {
            return _index(strides(), first, args_...);
        }

        template < size_t S >
        GT_FUNCTION int_t index(array< uint_t, S > const &a) const {
            return (int_t)explode< int_t, _impl_index >(a, *this);
        }

        /**@brief operator equals (same dimension size, etc.) */
        GT_FUNCTION
        constexpr bool operator==(meta_storage_base const &other) const {
            return (size() == other.size()) && (m_dims == other.m_dims) && (m_strides == other.m_strides);
        }
#else
        /**@brief straightforward interface*/
        GT_FUNCTION
        int_t index(uint_t const &i, uint_t const &j, uint_t const &k) const { return _index(strides(), i, j, k); }

        /**@brief operator equals (same dimension size, etc.) */
        GT_FUNCTION
        bool operator==(meta_storage_base const &other) const {
            return (size() == other.size()) && (m_dims == other.m_dims) && (m_strides == other.m_strides);
        }
#endif

        //####################################################
        // static functions (independent from the storage)
        //####################################################

        /**@brief helper code snippet to check if given vector coordinate is the maximum.*/
        template < uint_t Coordinate, typename T, typename Container >
        GT_FUNCTION static constexpr int_t get_stride_helper(Container const &cont, uint_t offset = 0) {
            typedef typename boost::mpl::deref<
                typename boost::mpl::max_element< typename T::layout_vector_t >::type >::type max_type;
            return (
                (max_type::value < 0) ? 0 : ((Layout::template at_< Coordinate >::value == max_type::value)
                                                    ? 1
                                                    : ((cont[Layout::template at_< Coordinate >::value + offset]))));
        }

        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively

           static version: the strides vector is passed from outside ordered in decreasing order, and the strides
           coresponding to
           the Coordinate dimension is returned according to the layout map.
           NOTE: the strides argument array contains only the strides and has dimension {space_dimensions-1}

           @tparam Coordinate the coordinate of which I want to retrieve the strides (0 for i, 1 for j, 2 for k)
           @tparam StridesVector the array type for the strides.
           @param strides_ the array of strides
        */
        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION static constexpr int_t strides(StridesVector const &RESTRICT strides_) {
            return get_stride_helper< Coordinate, layout >(strides_);
        }

        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively.

           non-static version.
        */
        template < uint_t Coordinate >
        GT_FUNCTION constexpr int_t strides() const {
            // NOTE: we access the m_strides vector starting from 1, because m_strides[0] is the total storage
            // dimension.
            return get_stride_helper< Coordinate, layout >(m_strides, 1);
        }

        /**@brief returning the index of the memory address corresponding to the specified (i,j,k) coordinates.
           This method depends on the strategy used (either naive or blocking). In case of blocking strategy the
           index for temporary storages is computed in the subclass gridtools::host_tmp_storge
           NOTE: this version will be preferred over the templated overloads
        */
        template < typename StridesVector >
        GT_FUNCTION static constexpr int_t _index(
            StridesVector const &RESTRICT strides_, uint_t const &i, uint_t const &j, uint_t const &k) {
            return strides_[0] * layout::template find_val< 0, uint_t, 0 >(i, j, k) +
                   strides_[1] * layout::template find_val< 1, uint_t, 0 >(i, j, k) +
                   layout::template find_val< 2, uint_t, 0 >(i, j, k);
        }

#ifdef CXX11_ENABLED
        /**
           @brief computing index to access the storage in the coordinates passed as parameters.

           This method must be called with integral type parameters, and the result will be a positive integer.
        */
        template < typename StridesVector, typename... UInt, typename Dummy = all_integers< UInt... > >
        GT_FUNCTION constexpr static int_t _index(StridesVector const &RESTRICT strides_, UInt const &... dims) {
            GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), boost::is_integral< UInt >::type::value...),
                "you have to pass in arguments of uint_t type");
            return _impl::compute_offset< space_dimensions, layout >::apply(strides_, dims...);
        }

#endif

        /**
           @brief computing index to access the storage in the coordinates passed as a tuple.

           \param StridesVector the vector of strides, it is a contiguous array of length space_dimenisons-1
           \param tuple is a tuple of coordinates, of type \ref gridtools::offset_tuple

           This method returns signed integers of type int_t (used e.g. in iterate_domain)
        */

        template < typename StridesVector, typename Offset >
        GT_FUNCTION static constexpr int_t _index(StridesVector const &RESTRICT strides_,
            Offset const &offset,
            typename boost::enable_if< typename is_tuple_or_array< Offset >::type, int >::type * = 0) {
            return _impl::compute_offset< space_dimensions, layout >::apply(strides_, offset);
        }

        template < typename StridesVector, typename LayoutT >
        GT_FUNCTION static constexpr int_t _index(
            StridesVector const &RESTRICT strides_, array< int_t, space_dimensions > const &offsets) {
            return _impl::compute_offset< space_dimensions, LayoutT >::apply(strides_, offsets);
        }

        template < typename OffsetTuple >
        GT_FUNCTION constexpr int_t _index(OffsetTuple const &tuple) const {
            GRIDTOOLS_STATIC_ASSERT((is_offset_tuple< OffsetTuple >::value), "wrong type");
            GRIDTOOLS_STATIC_ASSERT((space_dimensions <= layout::length), "something is very wrong");
            GRIDTOOLS_STATIC_ASSERT(OffsetTuple::n_dim > 0,
                "The placeholder is most probably not present in the domain_type you are using."
                " Double check that you passed all the placeholders in the domain_type construction");
            return _impl::compute_offset< space_dimensions, layout >::apply(strides(), tuple);
        }

        /** @brief returns the memory access index of the element with coordinate passed as an array

            \param StridesVector the vector of strides, it is a contiguous array of length space_dimenisons-1
            \param indices array of coordinates

            This method returns a signed int_t  (used e.g. in iterate_domain)*/
        template < typename IntType, typename StridesVector >
        GT_FUNCTION static constexpr int_t _index(StridesVector const &RESTRICT strides_, IntType *RESTRICT indices) {

            return _impl::compute_offset< space_dimensions, layout >::apply(strides_, indices);
        }

        /** @brief method to increment the memory address index by moving forward a given number of step in the given
           Coordinate direction
            \tparam Coordinate: the dimension which is being incremented (0=i, 1=j, 2=k, ...)
            \param steps: the number of steps of the increment
            \param index: the output index being set
        */
        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION static void increment(
            int_t const &steps_, int_t *RESTRICT index_, StridesVector const &RESTRICT strides_) {
// TODO assert(index_)
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(Coordinate < space_dimensions,
                "you have a storage in the iteration space whoose dimension is lower than the iteration space "
                "dimension. This might not be a problem, since trying to increment a nonexisting dimension has no "
                "effect. In case you want this feature comment out this assert.");

#endif
            if (layout::template at_< Coordinate >::value >= 0) // static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size() == space_dimensions - 1,
                    "error: trying to compute the storage index using strides from another storage which does not have "
                    "the same space dimensions. Are you explicitly incrementing the iteration space by calling "
                    "base_storage::increment?");
#endif
                *index_ += strides< Coordinate >(strides_) * steps_;
            }
        }

        /**
           @brief initializing a given coordinate (i.e. multiplying times its stride)

           \param steps_ the input coordinate value
           \param index_ the output index
           \param strides_ the strides array
         */
        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION static void initialize(uint_t const &steps_,
            uint_t const & /*block*/,
            int_t *RESTRICT index_,
            StridesVector const &RESTRICT strides_) {

            if (Coordinate < space_dimensions && layout::template at_< Coordinate >::value >= 0) // static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size() == space_dimensions - 1,
                    "error: trying to compute the storage index using strides from another storages which does not "
                    "have the same space dimensions. Sre you explicitly initializing the iteration space by calling "
                    "base_storage::initialize?");
#endif
                *index_ += strides< Coordinate >(strides_) * (steps_);
            }
        }

        /**
           returning 0 in a non blocked storage
        */
        GT_FUNCTION
        uint_t fields_offset(int_t EU_id_i, int_t EU_id_j) const { return 0; }
    };

} // namespace gridtools
