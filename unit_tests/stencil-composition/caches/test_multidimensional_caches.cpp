
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

namespace test_multidimensional_caches {
    using namespace gridtools;

    int test() {
#ifdef CUDA8
        typedef layout_map< 0, 1, 2, 3, 4, 5 > layout_t;
        typedef pointer< base_storage< wrap_pointer< double >, meta_storage_base< 0, layout_t, false >, 4 > > storage_t;
        typedef cache_storage<
            block_size< 8, 3, 4, 5, 6 >,
            extent< -1, 1, -2, 2, 0, 2, 0, 0, -1, 0 >,
            storage_t > cache_storage_t;
        typedef accessor< 0, enumtype::in, extent<>, 7 > acc_t;

        static constexpr cache_storage_t::meta_t m_;

        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(0) == 10, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(1) == 7, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(2) == 6, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(3) == 1, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(4) == 7, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.value().dim(5) == 4, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(1, 0, 0, 0, 0, 0)) == 1, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(2, 0, 0, 0, 0, 0)) == 2, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(3, 0, 0, 0, 0, 0)) == 3, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(4, 0, 0, 0, 0, 0)) == 4, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(5, 0, 0, 0, 0, 0)) == 5, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(6, 0, 0, 0, 0, 0)) == 6, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(7, 0, 0, 0, 0, 0)) == 7, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(8, 0, 0, 0, 0, 0)) == 8, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(9, 0, 0, 0, 0, 0)) == 9, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t{0, 0, 0, 0, 0, 0, 0}) == 0, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 1, 0, 0, 0, 0)) == 10, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 2, 0, 0, 0, 0)) == 20, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 3, 0, 0, 0, 0)) == 30, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 4, 0, 0, 0, 0)) == 40, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 5, 0, 0, 0, 0)) == 50, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 6, 0, 0, 0, 0)) == 60, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 1, 0, 0, 0)) == 70, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 2, 0, 0, 0)) == 140, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 3, 0, 0, 0)) == 210, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 4, 0, 0, 0)) == 280, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 5, 0, 0, 0)) == 350, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 1, 0, 0)) == 420, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 1, 0)) == 420, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 2, 0)) == 420 * 2, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 3, 0)) == 420 * 3, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 4, 0)) == 420 * 4, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 5, 0)) == 420 * 5, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 6, 0)) == 420 * 6, "error");

        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 0, 1)) == 420 * 7, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 0, 2)) == 420 * 7 * 2, "error");
        GRIDTOOLS_STATIC_ASSERT(m_.index(acc_t(0, 0, 0, 0, 0, 3)) == 420 * 7 * 3, "error");
#endif
        return 0;
    }
} // namespace test_multidimensional_caches

TEST(define_caches, test_sequence_caches) { test_multidimensional_caches::test(); }
