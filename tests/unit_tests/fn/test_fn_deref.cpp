#include <gridtools/fn/deref.hpp>

#include <memory>

#include <gtest/gtest.h>

namespace gridtools::fn {
    namespace {
        TEST(deref, ptr) {
            int val = 42;
            ASSERT_TRUE(can_deref(&val));
            EXPECT_EQ(deref(&val), 42);

            EXPECT_FALSE(can_deref(nullptr));
            EXPECT_FALSE(can_deref(0));
        }

        TEST(deref, smart_ptr) {
            auto p = std::make_unique<int>(42);
            ASSERT_TRUE(can_deref(p));
            EXPECT_EQ(deref(p), 42);
            p.reset();
            EXPECT_FALSE(p);
        }

        struct good {
            friend int fn_deref(good const &) { return 42; }
        };

        struct bad {
            friend bool fn_can_deref(bad const &) { return false; }
        };

        TEST(deref, smoke) {
            ASSERT_TRUE(can_deref(good()));
            EXPECT_EQ(deref(good()), 42);
            EXPECT_FALSE(can_deref(bad()));
        }
    } // namespace
} // namespace gridtools::fn
