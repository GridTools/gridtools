#pragma once
#include "../../storage/meta_storage_base.hpp"
#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools {

    template < typename Layout, uint_t... Dims >
    struct meta_storage_cache {

        typedef meta_storage_base< 0, Layout, false > meta_storage_t;

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
        GT_FUNCTION constexpr int_t const &strides() const {
            return m_value.template strides< Id >();
        }

        template < typename Accessor >
        GT_FUNCTION constexpr int_t index(Accessor const &arg_) const {
            return m_value._index(arg_);
        }

    };
} // namespace gridtools
