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

        using namespace literals;

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

            auto inc = [&] {
                sid::shift(ptr, sid::get_stride<D>(strides), Step());
                --count_down;
                ++count_up;
            };

            auto acc = tuple_util::fold(
                [&]<class Acc, class N>(Acc && acc, N) {
                    constexpr integral_constant<size_t, N::value> i;
                    auto res = Pass(i, count_down, std::forward<Acc>(acc), ptr, strides);
                    *at_key<Out>(ptr) = Get(acc);
                    inc();
                    return res;
                },
                Init(),
                meta::make_indices_c<PrologueSize, std::tuple>());

            for (; count_down >= EpilogueSize; inc())
                *at_key<Out>(ptr) = Get(acc = Pass(count_up, count_down, acc, ptr, strides));

            tuple_util::fold(
                [&]<class Acc, class N>(Acc && acc, N) {
                    constexpr integral_constant<size_t, EpilogueSize - 1 - N::value> i;
                    auto res = Pass(count_up, i, std::forward<Acc>(acc), ptr, strides);
                    *at_key<Out>(ptr) = Get(acc);
                    inc();
                    return res;
                },
                std::move(acc),
                meta::make_indices_c<EpilogueSize, std::tuple>());
        }

        template <auto Get, auto Pass, class In, class End, class Out, class T>
        void scan_body(In &in, End end, Out &out, T &acc) {
            for (; in != end; ++in)
                *out++ = Get(acc = Pass(acc, *in));
        }

        template <auto Prologue, auto Epilogue, auto Get, auto BodyPass, class In, class End, class Out>
        auto my_scan(In in, End end, Out out) {
            auto [acc, end2] = Prologue(in, end, out);
            scan_body<Get, BodyPass>(in, end2, out, acc);
            Epilogue(acc, in, end, out);
        }

        constexpr auto pass = []<class Num, class Acc, class T>(Num, Acc acc, T val) {
            if constexpr (is_integral_constant_of<Num, 0>())
                return std::tuple(val);
            else if constexpr (is_integral_constant_of<Num, 1>())
                return std::tuple(std::get<0>(acc), val);
            else if constexpr (is_integral_constant_of<Num, -2>())
                return std::tuple(std::get<1>(acc));
            else if constexpr (is_integral_constant_of<Num, -1>())
                return std::tuple();
            else
                return std::tuple(std::get<1>(acc), val);
        };

    } // namespace
} // namespace gridtools::fn
