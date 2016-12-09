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

namespace gridtools {
    /**
       @brief struct to allocate recursively a tuple of arrays (used to allocate the strides and dimensions of the
       storages)

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageList typelist of the storages
       \tparam ValueType the value (usually an integer) contained in the arrays
       \tparam Subtract an integer value: the allocated array will have dimension equal to the storage dimension minus
       this value. For instance when storing the strides, since we know that the smalles stride is 1, if the storage
       dimension is N we will store only N-1 strides in an array. In this case the value of the Subtract template
       parameter will be 1.
    */
    // TODOCOSUNA this is just an array, no need for special class, looks like
    template < ushort_t ID, typename MetaStorageList, typename ValueType, ushort_t Subtract >
    struct array_tuple : public array_tuple< ID - 1, MetaStorageList, ValueType, Subtract > {
        GRIDTOOLS_STATIC_ASSERT(boost::mpl::size< MetaStorageList >::value > ID,
            "Library internal error: strides index exceeds the number of storages");
        typedef typename boost::mpl::at_c< MetaStorageList, ID >::type storage_type;
        typedef array_tuple< ID - 1, MetaStorageList, ValueType, Subtract > super;
        typedef array< ValueType, storage_type::space_dimensions - Subtract > data_array_t;

#ifdef CXX11_ENABLED
        template < short_t Idx >
        using return_t = typename boost::mpl::if_< boost::mpl::bool_< Idx == ID >,
            data_array_t,
            typename super::template return_t< Idx > >::type;
#else
        template < short_t Idx >
        struct return_t {
            typedef typename boost::mpl::if_< boost::mpl::bool_< Idx == ID >,
                data_array_t,
                typename super::template return_t< Idx >::type >::type type;
        };
#endif

        /**@brief constructor, doing nothing more than allocating the space*/
        GT_FUNCTION
        array_tuple() : super() {}

        template < short_t Idx >
        GT_FUNCTION
#ifdef CXX11_ENABLED
            return_t< Idx >
#else
        typename return_t<Idx>::type
#endif
            const &RESTRICT get() const {
            return static_if< (Idx == ID) >::apply(m_data, super::template get< Idx >());
        }

        /**
           @bief getter for the i-th array
           \tparam Idx the index of the array in the tuple
         */
        template < short_t Idx >
        GT_FUNCTION
#ifdef CXX11_ENABLED
            return_t< Idx >
#else
        typename return_t<Idx>::type
#endif
                &RESTRICT get() {
            return static_if< (Idx == ID) >::apply(m_data, super::template get< Idx >());
        }

        GT_FUNCTION array_tuple(array_tuple const &other_) : super(other_), m_data(other_.template get< ID >()) {}

      private:
        data_array_t m_data;
    };

    /**specialization to stop the recursion*/
    template < typename MetaStorageList, typename ValueType, ushort_t Subtract >
    struct array_tuple< (ushort_t)0, MetaStorageList, ValueType, Subtract > {
        typedef typename boost::mpl::at_c< MetaStorageList, 0 >::type storage_type;

        GT_FUNCTION
        array_tuple() {}

        typedef array< ValueType, storage_type::space_dimensions - Subtract > data_array_t;

        template < short_t Idx >
#ifdef CXX11_ENABLED
        using return_t = data_array_t;
#else
        struct return_t {
            typedef data_array_t type;
        };
#endif

        template < short_t Idx >
        GT_FUNCTION data_array_t &RESTRICT get() { // stop recursion
            return m_data;
        }

        template < short_t Idx >
        GT_FUNCTION data_array_t const &RESTRICT get() const { // stop recursion
            return m_data;
        }

        GT_FUNCTION array_tuple(array_tuple const &other_) : m_data(other_.template get< 0 >()){};

      private:
        data_array_t m_data;
    };

    template < typename T >
    struct is_array_tuple : boost::mpl::false_ {};

    template < uint_t ID, typename StorageList, typename ValueType, ushort_t Subtract >
    struct is_array_tuple< array_tuple< ID, StorageList, ValueType, Subtract > > : boost::mpl::true_ {};

    /**
       @brief functor assigning the storage strides to the m_strides array.
       This is the unrolling of the inner nested loop

       @tparam BackendType the type of backend
       @tparam PEBlockSize the processing elements block size
    */
    template < typename BackendType, typename PEBlockSize, typename ValueType >
    struct assign_strides_inner_functor {
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");

      private:
        // while the strides are uint_t type in the storage metadata,
        // we stored them as int in the strides cached object in order to force vectorization
        ValueType *RESTRICT m_left;
        const ValueType *RESTRICT m_right;

      public:
        GT_FUNCTION
        assign_strides_inner_functor(ValueType *RESTRICT l, const ValueType *RESTRICT r) : m_left(l), m_right(r) {}

        template < typename ID >
        GT_FUNCTION void operator()(ID const &) const {
            assert(m_left);
            assert(m_right);
            // BackendType::template once_per_block< ID::value, PEBlockSize >::assign(
            //     m_left[ID::value], m_right[ID::value]);
            m_left[ID::value] = (ValueType)m_right[ID::value];
        }
    };

    /**@brief functor assigning the strides to a lobal array (i.e. m_strides).

       It implements the unrolling of a double loop: i.e. is n_f is the number of fields in this user function,
       and n_d(i) is the number of space dimensions per field (dependent on the ith field), then the loop for assigning
       the strides
       would look like
       for(i=0; i<n_f; ++i)
       for(j=0; j<n_d(i); ++j)
       * @tparam BackendType the type of backend
       * @tparam StridesCached strides cached type
       * @tparam MetaStorageSequence sequence of storages
       * @tparam PEBlockSize the processing elements block size
       */
    template < typename BackendType, typename StridesCached, typename MetaStorageSequence, typename PEBlockSize >
    struct assign_strides_functor {

        GRIDTOOLS_STATIC_ASSERT((is_array_tuple< StridesCached >::value), "internal error: wrong type");

      private:
        StridesCached &RESTRICT m_strides;
        const MetaStorageSequence &RESTRICT m_storages;
        assign_strides_functor();

      public:
        GT_FUNCTION
        assign_strides_functor(assign_strides_functor const &other)
            : m_strides(other.m_strides), m_storages(other.m_storages) {}

        GT_FUNCTION
        assign_strides_functor(StridesCached &RESTRICT strides, MetaStorageSequence const &RESTRICT storages)
            : m_strides(strides), m_storages(storages) {}

        template < typename Pair >
        GT_FUNCTION void operator()(Pair const &) const {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::second< Pair >::type::value <
                                        boost::fusion::result_of::size< MetaStorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");

            typedef typename boost::mpl::second< Pair >::type ID;

            typedef typename boost::mpl::first< Pair >::type meta_storage_type;

            // if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size< MetaStorageSequence >::value,
                "the ID is larger than the number of storage types");

#ifdef CXX11_ENABLED
#ifndef __CUDACC__
#if !defined(__INTEL_COMPILER)
            GRIDTOOLS_STATIC_ASSERT(
                (std::remove_reference< decltype(m_strides.template get< ID::value >()) >::type::size() ==
                    meta_storage_type::space_dimensions - 1),
                "internal error: the length of the strides vectors does not match. The bug fairy has no mercy.");
#endif
#endif
#endif
            boost::mpl::for_each< boost::mpl::range_c< short_t, 0, meta_storage_type::space_dimensions - 1 > >(
                assign_strides_inner_functor< BackendType, PEBlockSize, int_t >(
                    &(m_strides.template get< ID::value >()[0]),
                    &(boost::fusion::template at_c< ID::value >(m_storages)->strides(1))));
        }
    };

    template < typename BackendType, typename StridesCached, typename MetaStorageSequence, typename PEBlockSize >
    struct assign_dims_functor {

        GRIDTOOLS_STATIC_ASSERT((is_array_tuple< StridesCached >::value), "internal error: wrong type");

      private:
        StridesCached &RESTRICT m_dims;
        const MetaStorageSequence &RESTRICT m_storages;
        assign_dims_functor();

      public:
        GT_FUNCTION
        assign_dims_functor(assign_dims_functor const &other) : m_dims(other.m_dims), m_storages(other.m_storages) {}

        GT_FUNCTION
        assign_dims_functor(StridesCached &RESTRICT strides, MetaStorageSequence const &RESTRICT storages)
            : m_dims(strides), m_storages(storages) {}

        template < typename Pair >
        GT_FUNCTION void operator()(Pair const &) const {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::second< Pair >::type::value <
                                        boost::fusion::result_of::size< MetaStorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");

            typedef typename boost::mpl::second< Pair >::type ID;

            typedef typename boost::mpl::first< Pair >::type meta_storage_type;

            // if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size< MetaStorageSequence >::value,
                "the ID is larger than the number of storage types");

#ifdef CXX11_ENABLED
#ifndef __CUDACC__
#if !defined(__INTEL_COMPILER)
            GRIDTOOLS_STATIC_ASSERT(
                (std::remove_reference< decltype(m_dims.template get< ID::value >()) >::type::size() ==
                    meta_storage_type::space_dimensions),
                "internal error: the length of the strides vectors does not match. The bug fairy has no mercy.");
#endif
#endif
#endif
            boost::mpl::for_each< boost::mpl::range_c< short_t, 0, meta_storage_type::space_dimensions > >(
                assign_strides_inner_functor< BackendType, PEBlockSize, uint_t >(
                    &(m_dims.template get< ID::value >()[0]),
                    &(boost::fusion::template at_c< ID::value >(m_storages)->dims()[0])));
        }
    };

} // namespace gridtools
