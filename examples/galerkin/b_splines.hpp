#pragma once
#include "IGA/RunTimeIGA/b_splines_rt.h"
#include <boost/fusion/include/accumulate.hpp>
#include <tuple>

namespace gridtools{


    /**
       @brief struct containing ID and polynomial order
       \tparam I index
       \tparam P order
    */
    template<ushort_t I, ushort_t P>
    struct BSplineCoeff{
        static const ushort_t index=I;
        static const ushort_t order=P;
    };


    /** @brief just to ease the notation*/
    template <typename Coeff>
    struct BSplineDerived : iga_rt::BSpline<Coeff::order>
    {
        static const int index=Coeff::index;
        using super=iga_rt::BSpline<Coeff::order>;
        using super::BSpline;
    };

    template <typename Tuple>
    struct get_val
    {
    private:
        Tuple const& m_vals;
    public:
        get_val(Tuple const& vals_):
            m_vals(vals_)
        {}

        template<typename Basis>
        double operator()(Basis const& basis_) const
        {
            std::cout<<"Basis "<<Basis::index<<" for value "<<std::get<Basis::index>(m_vals)<<
                " returns "<<basis_.evaluate(std::get<Basis::index>(m_vals))<<std::endl;
            return basis_.evaluate(std::get<Basis::index>(m_vals));
        }
    };

    /**
       @brief b-spline function
    */
    template<typename ... Coeff>
    class GenericBSpline
    {
        using tuple_t=boost::fusion::vector<BSplineDerived<Coeff> ... >;

        tuple_t m_univariate_bsplines;

    public:

        template <typename ... Knots>
        GenericBSpline( Knots const& ... knots_ );

        /** @brief evaluates all the basis on the list of points*/
        template <typename ... Values>
        double evaluate(Values const& ... vals_) const{
            auto tuple_of_vals = boost::fusion::transform (
                m_univariate_bsplines
                , get_val<std::tuple<Values...> >(std::make_tuple(vals_...) ) );

            // std::cout<<"values of tuple: "<<
            //     boost::fusion::at_c<0>(tuple_of_vals)<<" "<<
            //     boost::fusion::at_c<1>(tuple_of_vals)<<" "<<
            //     boost::fusion::at_c<2>(tuple_of_vals)<<" "<<
            //     "size = "<<boost::fusion::size(tuple_of_vals)<<
            //     std::endl;

            //initial state
            double state(1.);

            return boost::fusion::accumulate(
                //calls the evaluate, and returns a sequence of evaluation
                tuple_of_vals
                , state
                , multiplies()
                ) ;
        }


    };

    //implementation
    template <typename ... Coeff>
    template <typename ... Knots>
    GenericBSpline<Coeff ...>::GenericBSpline( Knots const& ... knots_ ):
        m_univariate_bsplines(
            GenericBSpline<Coeff ...>::tuple_t(
                std::get<Coeff::index>(
                    std::make_tuple(&knots_ ...))...))
    {
    }

    template<ushort_t P, ushort_t Dim>
    struct parametric_space;

    template<ushort_t P>
    struct parametric_space<P, 3>{
        array<double, P+2> m_knots_i;
        array<double, P+2> m_knots_j;
        array<double, P+2> m_knots_k;
        //normalization
        double factor=(P+2)/2.;

        parametric_space(){
            for(int i=0; i< P+2; ++i)
            {
                m_knots_i[i]=((i)-factor)/factor;
                m_knots_j[i]=((i)-factor)/factor;
                m_knots_k[i]=((i)-factor)/factor;
            }
        }
    };

    /**
       @class class implementing the B-Splines elemental basis functions

       this class implements the interface of the Trilinos package Intrepid, and can
       be used from whithin the trilions framework to compute Jacobians and other
       elemental operations. It contains the evaluation of the basis functions and their
       derivatives in a set of quadrature nodes provided from outside. Works with hexahedral
       elements only.

       NOTE: the inheritance makes sure that the knot vectors get initialized
       before the b_splines object.
    */
    template<ushort_t P, ushort_t Dim>
    struct b_spline : parametric_space<P, Dim> {

    private:
        using spline_t = GenericBSpline<BSplineCoeff<0,P>, BSplineCoeff<1,P>, BSplineCoeff<2,P> >;
        //array<gridtools::array<double, P+1>, Dim> m_knots_i;
        spline_t m_basis;

    public:

        /** @brief constructor */
        b_spline( ) :
            parametric_space<P, Dim>()
            , m_basis(this->m_knots_i, this->m_knots_j, this->m_knots_k)
        {
        }

        template <typename Storage>
        void getDofCoords(Storage& /*s*/){
            // should return the knots coordinate
            // to be implemented
            assert(false);
        }

        int getCardinality() const
        {
            //returns the number of basis functions
            return P+1;
        }

        /**
           @brief compute the values of an operator on the basis functions, evaluate
           on quadrature points

           @tparam Storage arbitrary storage type for the output values
           @tparam Quad arbitrary storage type for the quadrature points
           (might differ from the previous one)
         */
        template <typename Storage, typename Quad>
        void getValues(Storage& storage_, Quad const& quad_points_, Intrepid::EOperator op) const
        {

            switch (op){
            case Intrepid::OPERATOR_VALUE :
                for (int i=0; i< P+1; ++i)
                {
                    for (int k=0; k< quad_points_.dimension(0); ++k)
                    {
                        std::cout<<"quad point: ("<<quad_points_(k, 0)<<", "<<quad_points_(k, 1)<<", "<<quad_points_(k, 2)<<")"<<std::endl;
                        std::cout<<"value (product of the three): "<<m_basis.evaluate(quad_points_(k, 0),quad_points_(k, 1),quad_points_(k, 2))<<std::endl;
                        // scalar basis functions evaluated in quad points
                        // now all the basis are the same. TODO: introduce traslations
                        storage_(i,k)=m_basis.evaluate(quad_points_(k, 0),quad_points_(k, 1),quad_points_(k, 2));
                    }
                } break;
                default:
                {
                    std::cout<<"Operator not supported"<<std::endl;
                    assert(false);
                }
            }
        }
    };



}//namespace gridtools
