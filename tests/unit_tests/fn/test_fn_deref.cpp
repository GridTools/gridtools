#include <gridtools/fn/builtins.hpp>

#include <memory>

#include <gtest/gtest.h>

#include <gridtools/sid/concept.hpp>

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

        template <auto Get,
            auto Pass,
            auto Init,
            size_t PrologueSize,
            size_t EpilogueSize,
            class D,
            class Step,
            class Out,
            class Ptr,
            class Strides>
        void do_scan(size_t size, Ptr &&ptr, Strides const &strides) {
            assert(size > PrologueSize + EpilogueSize);

            size_t count_up = 0;
            size_t count_down = size - 1;
            auto const &v_stride = sid::get_stride<D>(strides);

            auto next = [&](auto up, auto down, auto prev) {
                auto cur = Pass(up, down, std::move(prev), ptr, strides);
                *at_key<Out>(ptr) = Get(cur);
                sid::shift(ptr, v_stride, Step());
                --count_down;
                ++count_up;
                return cur;
            };

            auto prologue = [&](auto acc) {
                using indices_t = meta::make_indices_c<PrologueSize, std::tuple>;
                const auto f = [&]<class I>(auto acc, I) {
                    return next(integral_constant<size_t, I::value>(), count_down, std::move(acc));
                };
                return tuple_util::fold(f, std::move(acc), indices_t());
            };

            auto body = [&](auto acc) {
                while (count_down >= EpilogueSize)
                    acc = next(count_up, count_down, std::move(acc));
                return acc;
            };

            auto epilogue = [&](auto acc) {
                using indices_t = meta::make_indices_c<EpilogueSize, std::tuple>;
                const auto f = [&]<class I>(auto acc, I) {
                    return next(count_up, integral_constant<size_t, EpilogueSize - 1 - I::value>(), std::move(acc));
                };
                tuple_util::fold(f, std::move(acc), meta::make_indices_c<EpilogueSize, std::tuple>());
            };

            epilogue(body(prologue(Init())));
        }
    } // namespace
} // namespace gridtools::fn
