#include <cstddef>
#include <gridtools/fn/extents.hpp>

#include <memory>

#include <gtest/gtest.h>

namespace gridtools::fn {
    namespace {
        struct a {};
        struct b {};

        static_assert(is_extent<extent<a, 0, 0>>::value);
        static_assert(is_extent<extent<a, -1, 1>>::value);

        // TEST(extents, is_extent) {}
    } // namespace
} // namespace gridtools::fn
