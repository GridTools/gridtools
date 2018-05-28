#include <cmath>

#include <iterator>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/fusion/include/invoke.hpp>
#include <boost/fusion/include/std_tuple.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common/defs.hpp>
#include <common/functional.hpp>
#include <common/tuple_util.hpp>
#include <common/generic_metafunctions/for_each.hpp>
#include <common/generic_metafunctions/meta.hpp>

using namespace gridtools;

///////////////// HERE IS STENCIL COMPOSITION SPEC ///////////////////////////

/*
 * Concepts:
 *   Accessor - a pure functor that has a signatures compatible with `double(ptrdiff_t = 0, ptrdiff_t = 0)`
 *
 *   ElementaryStencil - a pure generic functor that takes accessors and returns double: `double(Accessor const&...)`
 *
 *   Evaluator - a functor that takes ElementaryStencil as a first arguments. Other parameters correspond to
 *   the arguments of that ElementaryStencil.
 *
 *   StencilComposition - a pure generic functor that takes Evaluator as a first parameter. Other parameters represent
 *   inputs of the StencilComposition. StencilComposition defines the way how the composition of stencils is expressed
 *   via composition of functions. The actual invocation of the composition (like `a(x, b(x, y))`) should be never
 *   written dirtectly but delegated to evaluator (like `eval(a, eval(b, x, y))`)
 */

struct lap_f {
    template < class In >
    double operator()(In const &in) const {
        return 4 * in() - (in(1, 0) + in(0, 1) + in(-1, 0) + in(0, -1));
    }
};

struct flx_f {
    template < class In, class Lap >
    double operator()(In const &in, Lap const &lap) const {
        double res = lap(1, 0) - lap();
        if (res * (in(1, 0) - in()) > 0)
            res = 0;
        return res;
    }
};

struct fly_f {
    template < class In, class Lap >
    double operator()(In const &in, Lap const &lap) const {
        double res = lap(0, 1) - lap();
        if (res * (in(0, 1) - in()) > 0)
            res = 0;
        return res;
    }
};

struct out_f {
    template < class In, class Flx, class Fly, class Coeff >
    double operator()(In const &in, Flx const &flx, Fly const &fly, Coeff const &coeff) const {
        return in() - coeff() * (flx() - flx(-1, 0) + fly() - fly(0, -1));
    }
};

struct diffusion_f {
    template < class Eval, class Input, class Coeff, class Lap >
    static auto helper(Eval const &eval, Input const &input, Coeff const &coeff, Lap const &lap)
        GT_AUTO_RETURN((eval(out_f{}, input, eval(flx_f{}, input, lap), eval(fly_f{}, input, lap), coeff)));

    template < class Eval, class Input, class Coeff >
    auto operator()(Eval const &eval, Input const &input, Coeff const &coeff) const
        GT_AUTO_RETURN((helper(eval, input, coeff, eval(lap_f{}, input))));
};

/////////////////// THE END OF STENCIL COMPOSITION SPEC /////////////////////

template < class T >
struct _2d {
    T x;
    T y;
};

using size_2d_t = _2d< size_t >;
using ptrdiff_2d_t = _2d< ptrdiff_t >;

struct range {
    ptrdiff_t begin;
    ptrdiff_t end;
};

using grid_t = _2d< range >;

template < class Impl, class Offset >
struct shifted_f {
    Impl m_impl;
    Offset m_offset;

    double operator()(ptrdiff_t i = 0, ptrdiff_t j = 0) const { return m_impl(i + m_offset.x, j + m_offset.y); }
    double &operator()(ptrdiff_t i = 0, ptrdiff_t j = 0) { return m_impl(i + m_offset.x, j + m_offset.y); }
};

template < class Impl >
shifted_f< Impl, ptrdiff_2d_t > shift(Impl const &impl, ptrdiff_t x, ptrdiff_t y) {
    return {impl, {x, y}};
}

template < class Container >
size_t size(Container const &container) {
    return std::end(container) - std::begin(container);
}

/*
 *  Concept of Container: STD Container of STD Container of double's, where the size of inner containers are equal
 *  to each other.
 */

template < class Container >
size_2d_t size_2d(Container const &container) {
    size_t x = size(container);
    size_t y = x ? size(*std::begin(container)) : 0;
    return {x, y};
}

size_2d_t size_2d(grid_t const &grid) {
    return {static_cast< size_t >(grid.x.end - grid.x.begin), static_cast< size_t >(grid.y.end - grid.y.begin)};
}

static double ignored;

template < class Container >
struct adapt_f {
    adapt_f(Container &container) : m_impl(container), m_size{size_2d(container)} {}
    double operator()(size_t i = 0, size_t j = 0) const {
        using std::begin;
        return i < m_size.x && j < m_size.y ? *(begin(*(begin(m_impl) + i)) + j) : NAN;
    }
    double &operator()(size_t i = 0, size_t j = 0) {
        using std::begin;
        return i < m_size.x && j < m_size.y ? *(begin(*(begin(m_impl) + i)) + j) : ignored;
    }
    Container &m_impl;
    size_2d_t m_size;
};

// Adapts Container to Accessor
template < class Impl >
adapt_f< Impl > adapt(Impl &impl) {
    return {impl};
}

// Models Container
class tmp_storage {
    size_2d_t m_size;
    std::unique_ptr< double[] > m_data;

    template < class T >
    struct stripe {
        T *m_cur;
        size_t m_size;

        stripe() = default;
        stripe(T *cur, size_t size) : m_cur(cur), m_size(size) {}

        T *begin() const { return m_cur; }
        T *end() const { return m_cur + m_size; }
    };

    template < class T >
    struct iterator
        : stripe< T >,
          boost::iterator_facade< iterator< T >, stripe< T >, boost::random_access_traversal_tag, stripe< T > > {
        using stripe< T >::stripe;

        stripe< T > dereference() const { return *this; }
        bool equal(iterator const &other) const { return this->m_cur == other.m_cur; }
        void increment() { this->m_cur += m_size; }
        void decrement() { this->m_cur -= m_size; }
        void advance(ptrdiff_t n) { this->m_cur += n * this->m_size; }
        ptrdiff_t distance_to(iterator const &other) const { return (other.m_cur - this->m_cur) / this->m_size; }
    };

  public:
    tmp_storage(size_2d_t size) : m_size(size), m_data(new double[size.x * size.y]) {}

    iterator< double > begin() { return {m_data.get(), m_size.y}; }
    iterator< double > end() { return {m_data.get() + m_size.y * m_size.x, m_size.y}; }
    iterator< const double > begin() const { return {m_data.get(), m_size.y}; }
    iterator< const double > end() const { return {m_data.get() + m_size.y * m_size.x, m_size.y}; }
};

// Spec for intermediate representation
namespace spec {
    struct out {};
    template < size_t >
    struct in {
        using type = in;
    };
    template < size_t >
    struct tmp {
        using type = tmp;
    };
    template < class... >
    struct args {
        using type = args;
    };
    template < class Fun, class Out, class Args >
    struct stage {
        using type = stage;
    };
    template < class... >
    struct stages {
        using type = stages;
    };
}

template < class Output, class... Inputs >
struct accessor_maker_f {
    ptrdiff_2d_t &m_offset;
    grid_t const &m_grid;
    std::vector< tmp_storage > &m_tmp_storages;
    Output const &m_output;
    std::tuple< Inputs const &... > m_inputs;

    template < class Acc >
    shifted_f< Acc, ptrdiff_2d_t & > bind_offset(Acc const &acc) const {
        return {acc, m_offset};
    }

    shifted_f< Output, ptrdiff_2d_t & > operator()(spec::out) const { return bind_offset(m_output); }

    template < size_t I, class Input = GT_META_CALL(meta::at_c, (meta::list< Inputs... >, I)) >
    shifted_f< Input, ptrdiff_2d_t & > operator()(spec::in< I >) const {
        return bind_offset(std::get< I >(m_inputs));
    }

    template < size_t I, class Tmp = shifted_f< adapt_f< tmp_storage >, ptrdiff_2d_t > >
    shifted_f< Tmp, ptrdiff_2d_t & > operator()(spec::tmp< I >) const {
        return bind_offset(shift(adapt(m_tmp_storages[I]), -m_grid.x.begin, -m_grid.y.begin));
    }
};

template < class Output, class... Inputs >
accessor_maker_f< Output, Inputs... > make_accessor_maker(ptrdiff_2d_t &offset,
    grid_t const &grid,
    std::vector< tmp_storage > &tmp_storages,
    Output const &output,
    std::tuple< Inputs const &... > const &inputs) {
    return {offset, grid, tmp_storages, output, inputs};
};

struct run_stencil_f {
    template < class Fun, class OutTag, class... ArgTags, class Output, class... Inputs >
    void operator()(spec::stage< Fun, OutTag, spec::args< ArgTags... > >,
        grid_t const &grid,
        std::vector< tmp_storage > &tmp_storages,
        Output &output,
        std::tuple< Inputs const &... > const &inputs) const {
        ptrdiff_2d_t offset;
        auto accessor_maker = make_accessor_maker(offset, grid, tmp_storages, output, inputs);
        auto out_acc = accessor_maker(OutTag{});
        auto in_accs = tuple_util::transform(accessor_maker, std::tuple< ArgTags... >{});
        Fun fun;
        for (offset.x = grid.x.begin; offset.x != grid.x.end; ++offset.x)
            for (offset.y = grid.y.begin; offset.y != grid.y.end; ++offset.y)
                out_acc() = boost::fusion::invoke(fun, in_accs);
    }
};

template < class >
class intermediate_f;

template < class... Funs, class... Outs, class... Args >
class intermediate_f< spec::stages< spec::stage< Funs, Outs, Args >... > > {
    using stages_t = spec::stages< spec::stage< Funs, Outs, Args >... >;

    grid_t m_grid;
    std::vector< tmp_storage > m_tmp_storages;

  public:
    intermediate_f(grid_t const &grid) : m_grid(grid) {
        for (size_t i = 0; i != sizeof...(Funs)-1; ++i)
            m_tmp_storages.emplace_back(size_2d(grid));
    }

    template < class Output, class... Inputs >
    void run(Output &output, Inputs const &... inputs) {
        auto fun = std::bind(run_stencil_f{},
            std::placeholders::_1,
            std::cref(m_grid),
            std::ref(m_tmp_storages),
            std::ref(output),
            std::tie(inputs...));
        host_for_each< stages_t >(std::move(fun));
    }
};

namespace meta_detail {

    struct graph_eval_f {
        template < class Fun, class... Args >
        meta::list< Fun, spec::args< Args... > > operator()(Fun const &, Args const &...) const;
    };

    template < class T >
    GT_META_DEFINE_ALIAS(to_input, spec::in, T::value);

    template < size_t Arity,
        class Spec,
        class Indices = GT_META_CALL(meta::make_indices_c, Arity),
        class Inputs = GT_META_CALL(meta::transform, (to_input, Indices)),
        class InputsTuple = GT_META_CALL(meta::rename, (std::tuple, Inputs)),
        class SpecArgs = GT_META_CALL(meta::push_front, (InputsTuple, graph_eval_f)) >
#if GT_BROKEN_TEMPLATE_ALIASES
    struct call_graph {
        using type = decltype(boost::fusion::invoke(std::declval< Spec >(), std::declval< SpecArgs >()));
    };
#else
    using call_graph = decltype(boost::fusion::invoke(std::declval< Spec >(), std::declval< SpecArgs >()));
#endif

#if GT_BROKEN_TEMPLATE_ALIASES
#define LAZY_FLATTEN_CALL_GRAPH_IMPL flatten_call_graph_impl
#else
    template < class >
    struct lazy_flatten_call_graph_impl;

    template < class T >
    using flatten_call_graph_impl = typename lazy_flatten_call_graph_impl< T >::type;

#define LAZY_FLATTEN_CALL_GRAPH_IMPL lazy_flatten_call_graph_impl
#endif

    template < class Leaf >
    struct LAZY_FLATTEN_CALL_GRAPH_IMPL {
        using type = meta::list< Leaf >;
    };

    template < class Fun, class ArgList >
    struct LAZY_FLATTEN_CALL_GRAPH_IMPL< meta::list< Fun, ArgList > > {
        using child_graphs = GT_META_CALL(meta::transform, (flatten_call_graph_impl, ArgList));
        using child_nodes = GT_META_CALL(meta::flatten, child_graphs);
        using this_node = meta::list< Fun, ArgList >;
        using type = GT_META_CALL(meta::push_back, (child_nodes, this_node));
    };

    template < class CallGraph,
        class Nodes = GT_META_CALL(flatten_call_graph_impl, CallGraph),
        class UniqueNodes = GT_META_CALL(meta::dedup, Nodes) >
    GT_META_DEFINE_ALIAS(
        flatten_call_graph, meta::filter, (meta::is_instantiation_of< meta::list >::apply, UniqueNodes));

    template < class T >
    GT_META_DEFINE_ALIAS(to_tmp, spec::tmp, T::value);

    template < class Nodes,
        class T,
        size_t N = meta::length< Nodes >::value,
        size_t I = GT_META_CALL(meta::st_position, (Nodes, T))::value >
    GT_META_DEFINE_ALIAS(node_to_tag2,
        meta::if_,
        (bool_constant< I == N >,
                             T,
                             GT_META_CALL(meta::if_, (bool_constant< I == N - 1 >, spec::out, spec::tmp< I >))));

    template < class Nodes,
        class T,
        size_t N = meta::length< Nodes >::value,
        size_t I = GT_META_CALL(meta::st_position, (Nodes, T))::value >
    GT_META_DEFINE_ALIAS(node_to_tag_impl,
        meta::if_,
        (bool_constant< I == N >,
                             T,
                             GT_META_CALL(meta::if_, (bool_constant< I == N - 1 >, spec::out, spec::tmp< I >))));

    template < class Nodes >
    struct node_to_tag {
        template < class T >
        GT_META_DEFINE_ALIAS(apply, node_to_tag_impl, (Nodes, T));
    };

    template < class Nodes >
    struct stage_spec {
        template < class Node,
            class Fun = GT_META_CALL(meta::first, Node),
            class OutTag = GT_META_CALL(node_to_tag_impl, (Nodes, Node)),
            class ArgList = GT_META_CALL(meta::second, Node),
            class ArgTags = GT_META_CALL(meta::transform, (node_to_tag< Nodes >::template apply, ArgList)) >
        GT_META_DEFINE_ALIAS(apply, spec::stage, (Fun, OutTag, ArgTags));
    };

    template < size_t Arity,
        class Spec,
        class CallGraph = GT_META_CALL(call_graph, (Arity, Spec)),
        class Nodes = GT_META_CALL(flatten_call_graph, CallGraph),
        class StageList = GT_META_CALL(meta::transform, (stage_spec< Nodes >::template apply, Nodes)) >
    GT_META_DEFINE_ALIAS(stage_specs, meta::rename, (spec::stages, StageList));
}

using meta_detail::stage_specs;

template < size_t Arity, class Spec, class Stages = GT_META_CALL(stage_specs, (Arity, Spec)) >
intermediate_f< Stages > make_computation(grid_t const &grid, Spec) {
    return {grid};
}

template < class Container >
grid_t make_grid(Container const &container) {
    auto size = size_2d(container);
    return {{0, static_cast< ptrdiff_t >(size.x)}, {0, static_cast< ptrdiff_t >(size.y)}};
}

// Implementation of nested call.
template < class Fun, class Indices, class... Args >
struct applied_impl_f;

template < class Fun, template < class... > class L, class... Is, class... Args >
struct applied_impl_f< Fun, L< Is... >, Args... > {
    Fun m_fun;
    std::tuple< Args... > m_args;
    double operator()(ptrdiff_t i = 0, ptrdiff_t j = 0) const {
        return m_fun(shift(std::get< Is::value >(m_args), i, j)...);
    }
};

template < class Fun, class... Args >
using applied_f = applied_impl_f< Fun, GT_META_CALL(meta::make_indices_c, sizeof...(Args)), Args... >;

struct fuse_eval_f {
    template < class Fun, class... Args >
    applied_f< Fun, Args... > operator()(Fun const &fun, Args const &... args) const {
        return {fun, {args...}};
    };
};

template < class Spec, class... Args >
auto make_fused(Spec const &spec, Args const &... args) GT_AUTO_RETURN((spec(fuse_eval_f{}, args...)));

const double PI = std::atan(1) * 4;

// input data is taken form interface1 example
struct input_f {
    size_t m_x;
    size_t m_y;
    double operator()(double i = 0, double j = 0) const {
        if (i < 0 || i >= m_x || j < 0 || j >= m_y)
            return NAN;
        double t = i / m_x + 1.5 * j / m_y;
        return 5 + 8 * (2 + std::cos(PI * t) + std::sin(2 * PI * t)) / 4;
    }
};
input_f make_input(size_t x, size_t y) { return {x, y}; }

namespace static_tests {

    using meta_detail::call_graph;
    using meta::list;
    using namespace spec;

    using expected_call_graph = list<                                       //
        out_f,                                                              //
        args< in< 0 >,                                                      //
            list< flx_f, args< in< 0 >, list< lap_f, args< in< 0 > > > > >, //
            list< fly_f, args< in< 0 >, list< lap_f, args< in< 0 > > > > >, //
            in< 1 >                                                         //
            >                                                               //
        >;                                                                  //

    using actual_call_graph = GT_META_CALL(call_graph, (2, diffusion_f));

    static_assert(std::is_same< actual_call_graph, expected_call_graph >{}, "");

    using expected_stages = stages<                                       //
        stage< lap_f, tmp< 0 >, args< in< 0 > > >,                        //
        stage< flx_f, tmp< 1 >, args< in< 0 >, tmp< 0 > > >,              //
        stage< fly_f, tmp< 2 >, args< in< 0 >, tmp< 0 > > >,              //
        stage< out_f, out, args< in< 0 >, tmp< 1 >, tmp< 2 >, in< 1 > > > //
        >;                                                                //

    using actual_stages = GT_META_CALL(stage_specs, (2, diffusion_f));

    static_assert(std::is_same< actual_stages, expected_stages >{}, "");
}

TEST(toy, diffusion) {
    constexpr auto size = size_2d_t{9, 10};
    double output_buf[size.x][size.y];
    auto output = adapt(output_buf);
    auto input = make_input(size.x, size.y);
    auto coeff = [](ptrdiff_t = 0, ptrdiff_t = 0) { return .025; };
    auto grid = make_grid(output_buf);
    diffusion_f diffusion;
    auto expected = make_fused(diffusion, input, coeff);
    auto testee = make_computation< 2 >(grid, diffusion);
    testee.run(output, input, coeff);

    for (size_t i = 0; i < size.x; ++i)
        for (size_t j; j < size.y; ++j)
            EXPECT_THAT(output(i, j), testing::NanSensitiveDoubleEq(expected(i, j)));
}
