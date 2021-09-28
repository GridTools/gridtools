#include <gridtools/fn/builtins.hpp>
#include <gridtools/fn/column.hpp>

#include <bits/ranges_base.h>
#include <memory>
#include <ranges>

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
            friend int fn_builtin(builtins::deref, good const &) { return 42; }
        };

        struct bad {
            friend bool fn_builtin(builtins::can_deref, bad const &) { return false; }
        };

        TEST(deref, smoke) {
            ASSERT_TRUE(can_deref(good()));
            EXPECT_EQ(deref(good()), 42);
            EXPECT_FALSE(can_deref(bad()));
        }

        TEST(ranges, smoke) {
            using namespace literals;

            EXPECT_EQ(plus(1, 2), 3);
            std::vector x = {1, -2, 3, 4, -3};
            auto in = std::views::all(x);
            for (auto &&x : if_(less(in, 0_c), 0_c, in)) {
                std::cout << x << std::endl;
            }
        }

        TEST(scan, smoke) {
            int in[] = {1, 2, 3, 4, 5, 6};
            std::tuple init = {0, 100};
            auto fun = [](auto acc, auto val) { return std::tuple(std::get<1>(acc), val); };
            std::vector<decltype(init)> out;
            std::inclusive_scan(std::begin(in), std::end(in), std::back_inserter(out), fun, init);
            for (auto &&[x, y] : out) {
                std::cout << x << ", " << y << std::endl;
            }
        }
    } // namespace
} // namespace gridtools::fn
