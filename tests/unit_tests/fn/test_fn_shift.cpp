#include <gridtools/fn/builtins.hpp>

#include <type_traits>

#include <gtest/gtest.h>

namespace gridtools::fn {
    namespace {
        template <auto...>
        struct dummy {};

        template <auto... Offsets, auto Off>
        constexpr dummy<Offsets..., Off> fn_builtin(builtins::shift, meta::val<Off>, dummy<Offsets...>) {
            return {};
        }

        template <auto... Offsets, auto Off0, auto Off1>
        constexpr dummy<Offsets..., Off0, Off1> fn_builtin(builtins::shift, meta::val<Off0, Off1>, dummy<Offsets...>) {
            return {};
        }

        inline constexpr auto zero = shift<>;
        inline constexpr auto sh = shift<1, 2, 3>;

        static_assert(std::is_same_v<std::decay_t<decltype(zero(dummy<42>{}))>, dummy<42>>);
        static_assert(std::is_same_v<decltype(sh(dummy<42>{})), dummy<42, 1, 2, 3>>);

        TEST(shift, default) {
            auto arr = std::array{41, 42, 43};
            EXPECT_EQ(shift<1>(arr), &arr[1]);
        }
    } // namespace
} // namespace gridtools::fn
