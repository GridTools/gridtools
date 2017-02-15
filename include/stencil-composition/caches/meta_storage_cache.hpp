#pragma once
#include "common/storage_info_interface.hpp"

#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools {

    template < typename Layout, uint_t... Dims >
    struct meta_storage_cache {

        typedef storage_info_interface< 0, Layout > meta_storage_t;

      private:
        const meta_storage_t m_value;

      public:
        GT_FUNCTION
        constexpr meta_storage_cache(meta_storage_cache const &other) : m_value{other.m_value} {};

        /**NOTE: the last 2 dimensions are Component and FD by convention*/
        GT_FUNCTION
        constexpr meta_storage_cache() : m_value{Dims...} {};

        GT_FUNCTION
        constexpr meta_storage_t value() const { return m_value; }

        GT_FUNCTION
        constexpr uint_t size() const { return m_value.size(); }

        template < ushort_t Id >
        GT_FUNCTION constexpr int_t const &stride() const {
            return m_value.template stride< Id >();
        }

        GT_FUNCTION constexpr nano_array<unsigned, Layout::length> const &strides() const {
            return m_value.template strides();
        }

        template < ushort_t Id >
        GT_FUNCTION constexpr int_t const &dim() const {
            return m_value.template dim< Id >();
        }

        template < unsigned N, typename OffsetTuple, typename... Offsets >
        GT_FUNCTION constexpr typename boost::enable_if_c<(N>0), int_t>::type
        get_index(OffsetTuple const& ot, Offsets... o) const {
            return get_index<N-1>(ot, o..., ot.template get<N-1>());
        }

        template < unsigned N, typename OffsetTuple, typename... Offsets >
        GT_FUNCTION constexpr typename boost::enable_if_c<(N==0), int_t>::type
        get_index(OffsetTuple const& ot, Offsets... o) const {
            return m_value.index(o..., 0, 0);
         }

         template < typename Accessor >
         GT_FUNCTION constexpr int_t index(Accessor const &arg_) const {
            typedef typename Accessor::offset_tuple_t OffsetTuple;
            return get_index<OffsetTuple::n_dim>(arg_.offsets());
         }

    };
} // namespace gridtools
