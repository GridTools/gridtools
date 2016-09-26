#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

namespace test_storage_info_gpu_using {

    using namespace gridtools;
    typedef layout_map< 0, 1, 2 > layout_t;
    typedef meta_storage< meta_storage_base< 0, layout_t, false > > meta_t;
    typedef storage< base_storage< hybrid_pointer< double >, meta_t > > storage_t;

    template < typename T >
    __global__ void set(T *st_) {

        for (int i = 0; i < 11; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 13; ++k)
                    // st_->fields()[0].out();
                    // printf("(*st_)(i,j,k) = %d", (*st_)(i,j,k));
                    (*st_)(i, j, k) = (double)i + j + k;
    }

    TEST(storage_info, test_pointer) {
        meta_t meta_(11, 12, 13);
        storage_t st_(meta_, 5.);
        st_.h2d_update();
        st_.clone_to_device();

        // clang-format off
    set<<<1,1>>>(st_.get_pointer_to_use());
        // clang-format on

        st_.d2h_update();
        // st_.print();

        bool ret = true;
        for (int i = 0; i < 11; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 13; ++k)
                    if (st_(i, j, k) != (double)i + j + k) {
                        ret = false;
                    }

        ASSERT_TRUE(ret);
    }

} // namespace test_storage_info_gpu_pointer
