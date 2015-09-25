#include <gridtools.h>
#include <storage/storage.h>
#include <stencil-composition/make_computation.h>
#include <stencil-composition/backend.h>
#include <type_traits>

using namespace gridtools;
using namespace expressions;

namespace b_spline{
// This is the definition of the special regions in the "vertical" direction
    typedef interval<level<0,-1>, level<1,-1> > x_interval;
    typedef interval<level<0,-2>, level<1,1> > axis;

    //index
    static enumtype::Dimension<1>::Index i;
    //position
    typedef enumtype::Dimension<3> xi_;
    typedef enumtype::Dimension<4> eta_;
    xi_::Index xi;
    eta_::Index eta;
    // static enumtype::Dimension<3>::Index eta;//quadrature points whithin the knot-span

    //P order Cox-De Bohr functions
    template<
             typename Dim, /*dimension being considered*/
             typename Knots,
             typename Input>
    struct cox_de_bohr{
        Dim::Index xi;
        int_t p;  /*order*/
        int_t g;  /*input point local index*/
        int_t id; /*knot-span index offset*/
        constexpr cox_de_bohr(int_t p_, int_t g_, int_t id_):
            p(p_),
            g(g_),
            id(id_)
            {}
        //current knot-span index is "i", while "g" is the local id of the input points whithin the ith knot-span
        auto expr =
            (Input(xi+g)-Knots(i+Id))/(Knots(i+(Id+1))-Knots()) *
            cox_de_bohr<Knots, Input>(p-1, g, id).expr +
            (Knots(i+(Id+P+1))-Input(xi+g)) /
            (Knots(i+(Id+P+1))-Knots(i+(Id+1))) *
            cox_de_bohr<In, Input>(p-1, g, id+1).expr;

        constexpr
    };

    //first order Cox-De Bohr functions
    template<typename Dim,
             typename In,
             typename Input>
    struct cox_de_bohr<Dim, In, Input>{
        static Dim::Index xi;
        int_t p;  /*order*/
        int_t g;  /*input point local index*/
        int_t id; /*knot-span index offset*/
        constexpr cox_de_bohr(int_t p_, int_t g_, int_t id_):
            p(p_),
            g(g_),
            id(id_)
            {}
        auto expr = (if_then_else((Input{xi+g} < Knots{i+(id+1)} && Input{xi+g} >= Knots{i+id}), 1. , 0.) );
    };

    // template<uint_t K, template <uint_t L> class Expression>
    // struct apply_expr{
    //     constexpr apply_expr(){}
    //     // static const typename std::underlying_type<decltype(Expression<k>::value)>::type value =  Expression<k>::value;
    //     using type=decltype(Expression<K>::value);
    //     static constexpr type value = Expression<K>::value;
    // };

    // template<template <uint_t K> class Expression>
    // struct apply_expr<0, Expression>{
    //     constexpr apply_expr(){}
    //     static constexpr float_type value=0.;
    // };

    // template<uint_t K, template <uint_t L> class Expression>
    // constexpr typename apply_expr<K, Expression>::type apply_expr<K, Expression>::value;

    // template<template <uint_t L> class Expression>
    // constexpr float_type apply_expr<0, Expression>::value;

    struct assemble{
        constexpr assemble(uint_t p_, uint_t q_, uint_t gx_, uint_t gy_, uint_t id_):
            p(p_),
            q(q_),
            gx(gx_),
            gy(gy_),
            id(id_)
            {}
        int_t p,q,gx,gy,id;

        // //basis function
        // template <int_t I>
    };

    //evaluate the B-spline in I
    template <int_t k>
    struct expr {
        using points=accessor<0>; // control points grid
        using knots=accessor<1>;  // nodes grid (grid in current space)
        using in=accessor<2>;     // vector of points to be interpolated (parametric coordinates)
        using out=accessor<3>;    // output surface
        using comp=Dimension<5>;
        using N=cox_de_bohr<Dimension<3>, in, knots>;
        using M=cox_de_bohr<Dimension<4>, in, knots>;

        constexpr expr(assemble& assemble_):
            a(assemble_)
            {}
        assemble const& a;

        auto value_xu = ((N(p, gx, p+k-1).expr * points(x(-p+k-1)) ) + expr<k-1>(a).value_xu);
        auto value_yu = ((N(p, gx, p+k-1).expr * points(x(-p+k-1),comp(1)) ) + expr<k-1>(a).value_yu);
        auto value_zu = ((N(p, gx, p+k-1).expr * points(x(-p+k-1), comp(2)) ) + expr<k-1>(a).value_zu);

        auto value_xv = ((M(p, gx, p+k-1).expr * points(y(-p+k-1)) ) + expr<k-1>(a).value_xv);
        auto value_yv = ((M(p, gx, p+k-1).expr * points(y(-p+k-1),comp(1)) ) + expr<k-1>(a).value_yv);
        auto value_zv = ((M(p, gx, p+k-1).expr * points(y(-p+k-1), comp(2)) ) + expr<k-1>(a).value_zv);
    };


    //evaluate the B-spline in I
    template <>
    struct expr<0> {
        constexpr expr(assemble& assemble_)
            {}

        auto value_xu =0;
        auto value_yu =0;
        auto value_zu =0;

        auto value_xv =0;
        auto value_yv =0;
        auto value_zv =0;
    };

    struct evaluation{

        using points=accessor<0>; // control points grid
        using knots=accessor<1>;  // nodes grid (grid in current space)
        using in=accessor<2>;     // vector of points to be interpolated (parametric coordinates)
        using out=accessor<3>;    // output surface
        using arg_list=boost::mpl::vector<points, knots, in, out> ;

        constexpr evaluation(uint_t p_, uint_t q_, uint_t gx_, uint_t gy_max_, uint_t id_, uint_t kmax_):
            p(p_),
            q(q_),
            gx(gx_),
            gy(gy_),
            id(id_)
            {}
        int_t p,q,gx,gy,id;

        //run the evaluation (constant time)
        template <typename Evaluation>
        static void Do(Evaluation const & eval, x_interval /*useless*/){
            decltype(apply_expr<Kmax, expr>::value)::to_string::apply();
            eval(out())    =eval(expr<p+1>(a).value_x);
            eval(out(z(1)))=eval(expr<p+1>(a).value);
            eval(out(z(2)))=eval(apply_expr<Kmax, expr_z>::value);
        }
    };


void test(int_t d1, int_t d2, int_t d3){

#ifdef CUDA_EXAMPLE
#define BACKEND backend<enumtype::Cuda, enumtype::Naive >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<enumtype::Host, enumtype::Block >
#else
#define BACKEND backend<enumtype::Host, enumtype::Naive >
#endif
#endif

    typedef gridtools::layout_map<1,0> layout_parametric;
    typedef gridtools::layout_map<2,1,0> layout_control_points;
    typedef gridtools::layout_map<3,2,1,0> layout_input;
    typedef gridtools::layout_map<4,3,2,1,0> layout_output;

    typedef gridtools::storage_info< layout_parametric> storage_parametric_info;
    typedef gridtools::storage_info< layout_control_points> storage_control_points_info;
    typedef gridtools::layout_map<0,layout_input> storage_input_info;
    typedef gridtools::layout_map<0,layout_output> storage_output_info;

    typedef gridtools::BACKEND::storage_type<float_type, storage_parametric_info >::type knots_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, storage_control_points_info >::type control_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, storage_input_info >::type input_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, storage_output_info >::type output_storage_type;

    storage_parametric_info knots_info(d1,d2);      //2D parametric space
    storage_control_points_info points_info(d1,d2,3);   //control points
    storage_input_info in_info(d1,d2,4,4);     //2D parametric evaluation grid with 4x4 points per knot-span (3rd dimension is the local numeration)
    storage_output_info out_info(d1,d2,4,4,3);  //resulting b-spline surface

    knots_storage_type knots(knots_info);      //2D parametric space
    control_storage_type points(points_info);   //control points
    input_storage_type in(in_info);     //2D parametric evaluation grid with 4x4 points per knot-span (3rd dimension is the local numeration)
    output_storage_type out(out_info);  //resulting b-spline surface

    typedef arg<0, knots_storage_type> p_knots;
    typedef arg<1, control_storage_type> p_points;
    typedef arg<2, input_storage_type> p_in;
    typedef arg<3, output_storage_type> p_out;
    typedef boost::mpl::vector<p_knots, p_points, p_in, p_out> accessor_list;
    domain_type<accessor_list> domain
        (boost::fusion::make_vector(&knots, &points, &in, &out));

    ushort_t halo_size=0;
    uint_t di[5] = {halo_size, halo_size, halo_size, d1-3, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-3, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    auto bspline_computation =
        make_computation<gridtools::BACKEND, layout_t>
        (
            //cubic b-splines, with 4 dofs per knot-span.
            //the iteration space loops on the knot-spans (index i),
            //while the inner loop on the points per knot span is unrolled below.
            make_mss(
            enumtype::execute<enumtype::forward>()
            , make_esf<b_spline::evaluate<3, 0, 4> >(p_points(), p_knots(), p_in(), p_out())
            , make_esf<b_spline::evaluate<3, 1, 4> >(p_points(), p_knots(), p_in(), p_out())
            , make_esf<b_spline::evaluate<3, 2, 4> >(p_points(), p_knots(), p_in(), p_out())
            , make_esf<b_spline::evaluate<3, 3, 4> >(p_points(), p_knots(), p_in(), p_out())
                ),
            domain, coords
            );

    bspline_computation->ready();
    bspline_computation->steady();
    bspline_computation->run();

}

}


// namespace Lagrange{

//     using enumtype::Dimension<0>::Index=i;
//     using enumtype::Dimension<1>::Index=x;

//     template<uint_t P, uint_t g, typename in>
//     struct cox_de_bohr{
//         static constexpr auto expr = (in{x+g}-in{})/(in{i+1}-in{})*cox_de_bohr<P-1, g> + (in{i+P+1}-in{x+g})/(in{i+P+1}-in{i+1})*cox_de_bohr<P-1, g+1>
//     };

//     //First order Lagrange polynomials
//     template<uint_t g, typename in>
//     struct cox_de_bohr<0, g>{
//     static constexpr auto expr = (in{x+g} < knots{i+1} && in{x+g} >= knots{}) ? 1. : 0.;
//     };

//     template <uint_t P, uint_t Id, uint_t Kmax>
//     struct evaluate{
//         using points=accessor<0>;
//         using knots=accessor<1>;
//         using in=accessor<2>;
//         using out=accessor<3>;

//         template <uint_t I>
//         using N=cox_de_bohr<P, I, in, knots>;

//         template<ushort_t k>
//         struct expr {
//             static constexpr auto value = (N<Id-P+ k -1>::expr * points(x(-P+k-1)) ) + expr<k-1>::value;
//         };

//         template<>
//         struct expr<0> {
//             static constexpr auto value=0;
//         };

//         template <typename Evaluation>
//         static void Do(Evaluation const & eval){
//             eval(out())=eval(expr<Kmax>::value);
//         }
//     };
//}
