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

        constexpr auto sum = reduce<std::plus(), 0>;
        constexpr auto dot = reduce<[](auto acc, auto l, auto r) { return acc + l * r; }, 0>;

        TEST(reduce, smoke) { EXPECT_EQ(sum(testee()), 4 + 5 + 6); }

        TEST(reduce, two_args) { EXPECT_EQ(dot(testee(), testee()), 16 + 25 + 36); }

        TEST(reduce, sparse) { EXPECT_EQ(dot(testee(), std::array{2, 2, 2, 2}), 4 * 2 + 5 * 2 + 6 * 2); }
    } // namespace
} // namespace gridtools::fn
