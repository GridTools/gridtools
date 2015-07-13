#pragma once
#define PEDANTIC_DISABLED
#include <gridtools.hpp>
#include <storage/storage.hpp>
#include <stencil-composition/make_computation.hpp>
#include <stencil-composition/backend.hpp>
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
    template<int_t P,  /*order*/
             int_t g,  /*input point local index*/
             int_t Id, /*knot-span index offset*/
             typename Dim, /*dimension being considered*/
             typename Knots,
             typename Input>
    struct cox_de_bohr{
        //current knot-span index is "i", while "g" is the local id of the input points whithin the ith knot-span
        static typename Dim::Index d;
        static auto constexpr expr =
            (Input{d+g}-Knots{i+Id})/(Knots{i+(Id+1)}-Knots{}) *
            cox_de_bohr<P-1, g, Id, Dim, Knots, Input>::expr +
            (Knots{i+(Id+P+1)}-Input{d+g}) /
            (Knots{i+(Id+P+1)}-Knots{i+(Id+1)}) *
            cox_de_bohr<P-1, g, Id+1, Dim, Knots, Input>::expr;
    };

    //first order Cox-De Bohr functions
    template<int_t g /*input point local index*/,
             int_t Id /*knot-span index*/,
             typename Dim,
             typename Knots,
             typename Input>
    struct cox_de_bohr<0, g, Id, Dim, Knots, Input>{
        static typename Dim::Index d;
        static constexpr auto expr = (if_then_else((Input{d+g} < Knots{i+(Id+1)} && Input{d+g} >= Knots{i+Id}), 1. , 0.) );
    };

    template<uint_t K1, uint_t K2, template <uint_t M, uint_t L> class Expression>
    struct apply_expr{
        constexpr apply_expr(){}
        // static const typename std::underlying_type<decltype(Expression<k>::value)>::type value =  Expression<k>::value;
        using type=decltype(Expression<K1, K2>::value);
        static constexpr type value = Expression<K1, K2>::value;
    };

    template<uint_t K1, template <uint_t L, uint_t K2> class Expression>
    struct apply_expr<K1, 0, Expression>{
        constexpr apply_expr(){}
        static constexpr float_type value=0.;
    };

    template<uint_t K2, template <uint_t K1, uint_t L> class Expression>
    struct apply_expr<0, K2, Expression>{
        constexpr apply_expr(){}
        static constexpr float_type value=0.;
    };

    template<uint_t K1, uint_t K2, template <uint_t M, uint_t L> class Expression>
    constexpr typename apply_expr<K1, K2, Expression>::type apply_expr<K1, K2, Expression>::value;

    template<uint_t K1,template <uint_t M, uint_t L> class Expression>
    constexpr float_type apply_expr<K1, 0, Expression>::value;

    template<uint_t K2,template <uint_t M, uint_t L> class Expression>
    constexpr float_type apply_expr<0, K2, Expression>::value;

    template <uint_t P, uint_t Q, uint_t gx, uint_t gy>
    struct evaluate{
        using points=accessor<0, range<0,0,0>, 3>; // control points grid
        using knots=accessor<1, range<0,0,0>, 2>;  // nodes grid (grid in current space)
        using in=accessor<2, range<0,0,0>, 4>;     // vector of points to be interpolated (parametric coordinates)

        using x=enumtype::Dimension<1>;
        using y=enumtype::Dimension<2>;
        using comp=enumtype::Dimension<5>;
        //basis function in x direction
        template <int_t I>
        using N=cox_de_bohr<P, gx, I, enumtype::Dimension<3>, in, knots>;

        //basis function in y direction
        template <int_t I>
        using M=cox_de_bohr<Q, gy, I, enumtype::Dimension<4>, in, knots>;


        // Computing the basis interpolation in the given point with coordinates (gx,gy) local to the knot span
        //inner loop, on k1
        template<uint_t k1, uint_t k2>
        struct inner_expr_x {
            template <uint_t U, uint_t V>
            using dummy=inner_expr_x<U,V>;
            static constexpr auto value = ((N<-P+ k1 -1>::expr * points(x(-P+k1-1), y(-Q+k2-1) ) ) + apply_expr<k1-1, k2, dummy>::value);
        };
        template<uint_t k1, uint_t k2>
        struct inner_expr_y {
            template <uint_t U, uint_t V>
            using dummy=inner_expr_y<U,V>;
            static constexpr auto value = ((N<-P+ k1 -1>::expr * points(x(-P+k1-1), y(-Q+k2-1), comp(1)) ) + apply_expr<k1-1, k2, dummy>::value);
        };
        template<uint_t k1, uint_t k2>
        struct inner_expr_z {
            template <uint_t U, uint_t V>
            using dummy=inner_expr_z<U,V>;
            static constexpr auto value = ((N<-P+ k1 -1>::expr * points(x(-P+k1-1), y(-Q+k2-1), comp(2)) )  + apply_expr<k1-1, k2, dummy>::value);
        };

        //external loop, on k2
        template< uint_t k1, uint_t k2 >
        struct expr_x{
            template <uint_t U, uint_t V>
            using dummy=expr_x<U,V>;
            static constexpr auto value = inner_expr_x<k1, k2>::value + apply_expr<k1, k2-1, dummy>::value;
        };

        template< uint_t k1, uint_t k2 >
        struct expr_y{
            template <uint_t U, uint_t V>
            using dummy=expr_y<U,V>;
            static constexpr auto value = inner_expr_y<k1, k2>::value + apply_expr< k1, k2-1, dummy>::value;
        };

        template< uint_t k1, uint_t k2 >
        struct expr_z{
            template <uint_t U, uint_t V>
            using dummy=expr_z<U,V>;
            static constexpr auto value = inner_expr_z<k1, k2>::value + apply_expr< k1, k2-1, dummy>::value;
        };

    };

    template <int_t P, int_t Q>
    struct compute_functor{

        using points=accessor<0, range<0,0,0>, 3>; // control points grid
        using knots=accessor<1, range<0,0,0>, 2>;  // nodes grid (grid in current space)
        using in=accessor<2, range<0,0,0>, 4>;     // vector of points to be interpolated (parametric coordinates)
        using out=accessor<3, range<0,0,0>, 5>;    // output surface
        using z=enumtype::Dimension<3>;

        template <typename Evaluation, typename GX >
        struct inner_loop_functor{
        private:
            Evaluation const& m_eval;
        public:
            constexpr inner_loop_functor(Evaluation const& eval_):
                m_eval(eval_){}

            template <typename GY>
            void operator() (GY /**/){
                using evaluate_t=evaluate<P,Q,GX::value,GY::value>;
                m_eval(out())    =m_eval(apply_expr<P+1, Q+1, evaluate_t::template expr_x>::value);
                m_eval(out(z(1)))=m_eval(apply_expr<P+1, Q+1, evaluate_t::template expr_y>::value);
                m_eval(out(z(2)))=m_eval(apply_expr<P+1, Q+1, evaluate_t::template expr_z>::value);
            }
        };

        template <typename Evaluation, int_t GY>
        struct outer_loop_functor{
        private:
            Evaluation const& m_eval;
        public:
            constexpr outer_loop_functor(Evaluation const& eval_):
                m_eval(eval_){}

            template<typename GX>
            void operator() (GX /**/){
                for_each<boost::mpl::range_c<uint_t, 0,GY> >(inner_loop_functor<Evaluation, GX>(m_eval));
            }
        };

        template <int_t GX, int_t GY>
        struct eval_functor{

            using arg_list=boost::mpl::vector<points, knots, in, out> ;

            // static void print(){
            //     decltype(apply_expr<P+1, Q+1, evaluate<P,Q,GX,GY>::template expr_x>::value)::to_string::apply();
            // }

            //run the evaluation (constant time)
            template <typename Evaluation>
            static void Do(Evaluation const & eval, x_interval /*useless*/){
                for_each<boost::mpl::range_c<uint_t,0,GX> >(outer_loop_functor<Evaluation, GY>(eval));

            }
        };
    };

void test(uint_t d1, uint_t d2, uint_t d3){

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
    typedef gridtools::BACKEND::storage_type<float_type, layout_parametric >::type knots_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, layout_control_points >::type control_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, layout_input >::type input_storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, layout_output >::type output_storage_type;

    knots_storage_type knots(d1,d2);      //2D parametric space
    control_storage_type points(d1,d2,3);   //control points
    input_storage_type in(d1,d2,4,4);     //2D parametric evaluation grid with 4x4 points per knot-span (3rd dimension is the local numeration)
    output_storage_type out(d1,d2,4,4,3);  //resulting b-spline surface

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
        make_computation<gridtools::BACKEND, layout_control_points>
        (
            //cubic b-splines, with 16 dofs per knot-span.
            //the iteration space loops on the knot-spans (index i),
            //while the inner loop on the points per knot span is unrolled below.
            make_mss(
                enumtype::execute<enumtype::forward>()//all possible 4x4 combinations
                , make_esf<compute_functor<1,1>::eval_functor<1,1> >(p_points(), p_knots(), p_in(), p_out())
                ),
            domain, coords
            );

    bspline_computation->ready();
    bspline_computation->steady();
    bspline_computation->run();

}

}
