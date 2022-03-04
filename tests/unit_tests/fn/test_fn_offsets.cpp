#include <gridtools/fn/offsets.hpp>

#include <functional>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/fn/builtins.hpp>

namespace gridtools::fn {
    namespace {
        constexpr inline int data[] = {4, 5, 100, 6};

        using indices_t = std::tuple<integral_constant<int, 0>,
            integral_constant<int, 1>,
            integral_constant<int, -1>,
            integral_constant<int, 3>>;

        struct testee {
            template <auto I>
            friend constexpr int const *fn_builtin(builtins::shift<I>, testee) {
                return data + I;
            }
            friend constexpr bool fn_builtin(builtins::can_deref, testee) { return false; }
            friend constexpr indices_t fn_offsets(testee) { return {}; }
        };

        constexpr inline auto offs = offsets(testee());
        static_assert(std::is_same_v<decltype(offs), indices_t const>);

        constexpr std::plus plus;
        constexpr auto zero = [](...) { return 0; };
        constexpr auto sum = reduce<plus, zero>;
        constexpr auto dot_helper = [](auto acc, auto l, auto r) { return acc + l * r; };
        constexpr auto dot = reduce<dot_helper, zero>;

        TEST(reduce, smoke) { EXPECT_EQ(sum(testee()), 4 + 5 + 6); }

        TEST(reduce, two_args) { EXPECT_EQ(dot(testee(), testee()), 16 + 25 + 36); }
    } // namespace
} // namespace gridtools::fn
