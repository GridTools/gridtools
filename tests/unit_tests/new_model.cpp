#include <functional>
#include <optional>
#include <tuple>

#include <gtest/gtest.h>

#include <gridtools/common/compose.hpp>
#include <gridtools/common/for_each.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/stencil/common/dim.hpp>

using namespace gridtools;
using namespace literals;

inline constexpr std::plus plus{};
inline constexpr std::minus minus{};
inline constexpr std::multiplies multiplies{};
inline constexpr std::divides divides{};

inline constexpr auto shift() {
    return [](auto it) { return it; };
}

constexpr auto shift(auto d, auto... offsets) {
    return [s = shift(offsets...), d](auto it) { return s(shift_impl(d, it)); };
}

constexpr auto shift(auto d, auto val, auto... offsets) {
    return [s = shift(offsets...), d, val](auto it) { return s(shift_impl(d, val, it)); };
}

template <class Key, class Ptr, class Strides>
struct strided_iter {
    Ptr ptr;
    Strides const &strides;

    friend strided_iter shift_impl(auto d, auto val, strided_iter it) {
        sid::shift(it.ptr, sid::get_stride_element<Key, decltype(d)>(it.strides), val);
        return it;
    }

    friend auto deref(strided_iter const &it) { return *it.ptr; }
};

template <class Stencil, class Args>
struct lifted_iter {
    Stencil stencil;
    Args args;

    lifted_iter(Stencil stencil, Args args) : stencil(stencil), args(args) {}

    friend auto shift_impl(auto d, auto val, lifted_iter it) {
        return lifted_iter(it.stencil, tuple_util::transform(shift(d, val), it.args));
    }

    friend auto deref(lifted_iter const &it) { return std::apply(it.stencil, it.args); }
};

constexpr auto ilift = [](auto stencil) {
    return [=](auto... its) {
        using res_t = decltype(stencil(its...));
        if constexpr (meta::is_instantiation_of<std::tuple, res_t>())
            return tuple_util::transform(
                [=](auto i) {
                    return lifted_iter(
                        [=](auto... its) { return std::get<decltype(i)::value>(stencil(its...)); }, std::tuple(its...));
                },
                meta::rename<std::tuple, meta::make_indices<tuple_util::size<res_t>>>());
        else
            return lifted_iter(stencil, std::tuple(its...));
    };
};

struct cartesian {
    int i[2];
    int j[2];
    int k[2];
};

namespace dim = stencil::dim;

inline constexpr dim::i i{};
inline constexpr dim::j j{};
inline constexpr dim::k k{};

template <class T>
struct out_tag : T {};

template <class T>
struct in_tag : T {};

template <class Src>
auto to_tuple(Src &&src) {
    if constexpr (meta::is_instantiation_of<std::tuple, std::decay_t<Src>>())
        return std::forward<Src>(src);
    else
        return std::tuple<std::decay_t<Src>>{std::forward<Src>(src)};
}

template <class Stencil, class Outputs, class Inputs>
void apply_stencil(cartesian domain, Stencil const &stencil, Outputs const &outputs, Inputs const &inputs) {
    using out_tags_t = meta::transform<out_tag, meta::make_indices<tuple_util::size<Outputs>>>;
    using in_tags_t = meta::transform<in_tag, meta::make_indices<tuple_util::size<Inputs>>>;
    using keys_t = meta::rename<sid::composite::keys, meta::concat<out_tags_t, in_tags_t>>;
    auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::concat(outputs, inputs));

    auto strides = sid::get_strides(composite);
    auto ptr = sid::get_origin(composite)();

    sid::shift(ptr, sid::get_stride<dim::i>(strides), domain.i[0]);
    sid::shift(ptr, sid::get_stride<dim::j>(strides), domain.j[0]);
    sid::shift(ptr, sid::get_stride<dim::k>(strides), domain.k[0]);

    compose(sid::make_loop<dim::k>(domain.i[1] - domain.i[0]),
        sid::make_loop<dim::j>(domain.j[1] - domain.j[0]),
        sid::make_loop<dim::k>(domain.k[1] - domain.k[0]))([&](auto &ptr, auto const &strides) {
        auto srcs = to_tuple(std::apply(stencil,
            tuple_util::transform(
                [&](auto tag) {
                    using tag_t = decltype(tag);
                    auto p = at_key<decltype(tag)>(ptr);
                    return strided_iter<tag_t, std::decay_t<decltype(p)>, std::decay_t<decltype(strides)>>{
                        at_key<decltype(tag)>(ptr), strides};
                },
                meta::rename<std::tuple, in_tags_t>())));
        for_each<out_tags_t>([&](auto tag) {
            using tag_t = decltype(tag);
            *at_key<tag_t>(ptr) = std::get<tag_t::value>(srcs);
        });
    })(ptr, strides);
}

template <class Domain, class Stencil, class Outputs, class... Inputs>
struct closure {
    Domain domain;
    Stencil stencil;
    Outputs outputs;
    std::tuple<Inputs const &...> inputs;

    closure(Domain domain, Stencil stencil, Outputs outputs, Inputs const &... inputs)
        : domain(domain), stencil(stencil), outputs(outputs), inputs(inputs...) {}
};

auto out(auto &... args) { return std::tie(args...); }

void fencil(auto... closures) {
    (..., (apply_stencil(closures.domain, closures.stencil, closures.outputs, closures.inputs)));
}

inline constexpr auto lift = ilift;

/// lap stencils

inline constexpr auto ldif = [](auto d) {
    return [s = shift(d, -1_c)](auto in) { return minus(deref(s(in)), deref(in)); };
};

inline constexpr auto rdif = [](auto d) { return compose(ldif(d), shift(d, 1_c)); };

inline constexpr auto dif2 = [](auto d) { return compose(ldif(d), lift(rdif(d))); };

inline constexpr auto lap = [](auto in) { return plus(dif2(i)(in), dif2(j)(in)); };

template <Sid In, Sid Out>
void testee(cartesian domain, Out &output, In &input) {
    fencil(closure(domain, lap, out(output), input));
}

//////////

TEST(carthesian, lap) {
    double actual[10][10][3] = {};
    double in[10][10][3] = {};
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 10; ++k)
                in[i][j][k] = i + j + k;

    constexpr cartesian domain = {.i = {1, 9}, .j = {1, 9}, .k = {0, 3}};

    constexpr auto expected = [](auto in, auto i, auto j, auto k) {
        return 4 * in[i][j][k] - in[i + 1][j][k] - in[i - 1][j][k] - in[i][j + 1][k] - in[i][j - 1][k];
    };

    testee(domain, actual, in);

    for (int i = domain.i[0]; i < domain.i[1]; ++i)
        for (int j = domain.j[0]; j < domain.j[1]; ++j)
            for (int k = domain.k[0]; k < domain.k[1]; ++k)
                EXPECT_DOUBLE_EQ(actual[i][j][k], expected(in, i, j, k)) << i << j << k;
}

template <auto>
struct connectivity {};

inline constexpr struct horizontal_t {
} horizontal;

template <class Indices, class StridedIter>
struct neighbors_iter {
    Indices indices;
    StridedIter impl;

    friend std::optional<decltype(deref(std::declval<StridedIter const &>()))> deref(neighbors_iter const &it) {
        return {};
    }
};

template <class StridedIter>
struct neighbor_iter {
    int index;
    StridedIter impl;

    template <auto Table>
    friend auto shift_impl(connectivity<Table>, neighbor_iter const &it) {
        return neighbors_iter(Table[it.index], it.impl);
    }

    friend neighbor_iter shift_impl(auto d, auto val, neighbor_iter it) {
        it.impl = shift(d, val)(it);
        return it;
    }

    friend std::optional<decltype(deref(std::declval<StridedIter const &>()))> deref(neighbor_iter const &it) {
        if (it.index == -1)
            return {};
        return deref(shift(horizontal, it.index)(it.impl));
    }
};

template <class Indices, class StridedIter>
neighbor_iter<StridedIter> shift_impl(auto val, neighbors_iter<Indices, StridedIter> it) {
    return {it.indices[val], it.impl};
}

constexpr auto reduce(auto fun, auto init) {
    return [=](auto const &arg, auto const &... args) {
        auto res = init;
        for (int i : arg.indices)
            if (i != -1)
                res = std::apply(fun,
                    std::tuple(res, deref(shift(horizontal, i)(arg.impl)), deref(shift(horizontal, i)(args.impl))...));
        return res;
    };
}

/*
 *          (2)
 *       1   2    3
 *   (1)  0     4   (3)
 *   11     (0)      5
 *   (6) 10      6  (4)
 *      9    8   7
 *          (5)
 *
 */

inline constexpr int e2v_table[12][2] = {
    {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 0}, {3, 5}, {4, 0}, {4, 5}, {5, 0}, {5, 6}, {6, 0}, {6, 1}};

inline constexpr int v2e_table[7][6] = {{0, 2, 4, 6, 8, 10},
    {0, 1, 11, -1, -1, -1},
    {1, 2, 3, -1, -1, -1},
    {3, 4, 5, -1, -1, -1},
    {5, 6, 7, -1, -1, -1},
    {7, 8, 9, -1, -1, -1},
    {9, 10, 11, -1, -1, -1}};

inline constexpr connectivity<e2v_table> e2v;
inline constexpr connectivity<v2e_table> v2e;

inline constexpr auto sum = reduce(plus, 0);
inline constexpr auto dot = reduce([](auto acc, auto x, auto y) { return acc + x * y; }, 0);

inline constexpr auto zavg = [](auto pp, auto sx, auto sy) {
    auto tmp = mult(.5, reduce(plus, 0)(shift(e2v)(pp)));
    return std::tuple(mult(tmp, *deref(sx)), mult(tmp, *deref(sy)));
};

inline constexpr auto nabla = [](auto pp, auto sx, auto sy, auto sign, auto vol) {
    auto [x, y] = lift(zavg)(pp, sx, sy);
    return std::tuple(divides(dot(shift(v2e)(x), sign)), divides(dot(shift(v2e)(x), sign)));
};
