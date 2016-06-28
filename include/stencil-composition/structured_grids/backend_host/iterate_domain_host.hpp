/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include "stencil-composition/iterate_domain_fwd.hpp"
#include "stencil-composition/iterate_domain.hpp"
#include "stencil-composition/iterate_domain_metafunctions.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the Host backend
     */
    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    class iterate_domain_host
        : public IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_host);
        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal error: wrong type");

        typedef IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > super;

        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename super::reduction_type_t reduction_type_t;

      public:
        using super::operator();
        typedef iterate_domain_host iterate_domain_t;
        typedef typename super::data_pointer_array_t data_pointer_array_t;
        typedef typename super::strides_cached_t strides_cached_t;
        typedef boost::mpl::map0<> ij_caches_map_t;

        GT_FUNCTION
        explicit iterate_domain_host(
            local_domain_t const &local_domain, const reduction_type_t &reduction_initial_value)
            : super(local_domain, reduction_initial_value), m_data_pointer(0), m_strides(0) {}

        void set_data_pointer_impl(data_pointer_array_t *RESTRICT data_pointer) {
            assert(data_pointer);
            m_data_pointer = data_pointer;
        }

        data_pointer_array_t &RESTRICT data_pointer_impl() {
            assert(m_data_pointer);
            return *m_data_pointer;
        }
        data_pointer_array_t const &RESTRICT data_pointer_impl() const {
            assert(m_data_pointer);
            return *m_data_pointer;
        }

        strides_cached_t &RESTRICT strides_impl() {
            assert(m_strides);
            return *m_strides;
        }

        strides_cached_t const &RESTRICT strides_impl() const {
            assert(m_strides);
            return *m_strides;
        }

        void set_strides_pointer_impl(strides_cached_t *RESTRICT strides) {
            assert(strides);
            m_strides = strides;
        }

        iterate_domain_host const& get() const {return *this;}

        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment_impl() {}

        template < ushort_t Coordinate >
        GT_FUNCTION void increment_impl(int_t steps) {}

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize_impl() {}

        template < typename ReturnType, typename Accessor, typename StoragePointer >
        GT_FUNCTION ReturnType get_value_impl(
            StoragePointer RESTRICT &storage_pointer, const uint_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");

            return super::template get_gmem_value< ReturnType >(storage_pointer, pointer_offset);
        }

        //    template <typename MetaDataSequence, typename ArgStoragePair0, typename... OtherArgs>
        //    typename boost::enable_if_c< is_any_storage<typename ArgStoragePair0::storage_type>::type::value
        //                                , void>::type assign_pointers

        //    typename boost::enable_if<MultipleGridPointsPerWarp, int >::type=0

      private:
        data_pointer_array_t *RESTRICT m_data_pointer;
        strides_cached_t *RESTRICT m_strides;
    };

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain_host< IterateDomainBase, IterateDomainArguments > >
        : public boost::mpl::true_ {};

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_positional_iterate_domain< iterate_domain_host< IterateDomainBase, IterateDomainArguments > >
        : is_positional_iterate_domain<
              IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > > {};

} // namespace gridtools
