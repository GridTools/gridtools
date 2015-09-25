#pragma once
#include "IGA/RunTimeIGA/b_splines_rt.h"
#include <boost/fusion/include/accumulate.hpp>
#include <tuple>
#include "nest_loops.hpp"
#include <common/generic_metafunctions/gt_integer_sequence.hpp>
// #include <storage/meta_storage_base.hpp>

namespace gridtools{


    /**
       @brief struct containing ID and polynomial order
       \tparam Dim current dimension (for multivariate)
       \tparam I index
       \tparam P order
    */
    template<ushort_t Dim, ushort_t P, ushort_t I>
    struct BSplineCoeff{
        static const ushort_t dimension=Dim;
        static const ushort_t order=P;
        static const ushort_t index=I;
    };


    /** @brief just to ease the notation*/
    template <typename Coeff>
    struct BSplineDerived : iga_rt::BSpline<Coeff::index, Coeff::order>
    {
        static const int index=Coeff::index;
        static const int dimension=Coeff::dimension;
        using super=iga_rt::BSpline<Coeff::index,Coeff::order>;
        //inheriting constructors
        using iga_rt::BSpline<Coeff::index,Coeff::order>::BSpline;
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
            // std::cout<<"Basis "<<Basis::dimension<<" for value "<<std::get<Basis::dimension>(m_vals)<<
            //     " returns "<<basis_.evaluate(std::get<Basis::dimension>(m_vals))<<std::endl;
            return basis_.evaluate(std::get<Basis::dimension>(m_vals));
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

        // template <typename ... Knots>
        // GenericBSpline( Knots const& ... knots_ );
        template <typename Knots>
        GenericBSpline( Knots const& knots_ );

        /** @brief evaluates all the basis on the list of points*/
        template <typename ... Values>
        double evaluate(Values const& ... vals_) const{
            //check length vals_ == length m_univariate_bsplines
            auto tuple_of_vals = boost::fusion::transform (
                m_univariate_bsplines
                , get_val<std::tuple<Values...> >(std::make_tuple(vals_...) ) );

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

    // //implementation
    // template <typename ... Coeff>
    // template <typename ... Knots>
    // GenericBSpline<Coeff ...>::GenericBSpline( Knots const& ... knots_ ):
    //     //check that sizeof...(Knots) is same as Coeff
    //     m_univariate_bsplines(
    //         GenericBSpline<Coeff ...>::tuple_t(
    //             std::get<Coeff::dimension>(
    //                 std::make_tuple(&knots_ ...))...))
    // {
    // }

    //implementation
    template <typename ... Coeff>
    template <typename Knots>
    GenericBSpline<Coeff ...>::GenericBSpline( Knots const& knots_ ):
        //check that sizeof...(Knots) is same as Coeff
        m_univariate_bsplines(
            GenericBSpline<Coeff ...>::tuple_t(
                &knots_[Coeff::dimension]...))
    {
    }

    template<ushort_t P, ushort_t Dim>
    struct parametric_space{
	array<array<double, P+P+1>, Dim> m_knots;

        parametric_space()
	{
	    int k=0;
	    for(int i=0; i< (P+P+1)*2; i+=2)
	    {
#ifdef VERBOSE
		std::cout<<" knots at "<<(double)((i)+1.-2.*P)<<std::endl;
#endif
		for(int d=0; d<Dim; d++)
		{
		    m_knots[d][k]=(double)((i)+1.-2.*P);///factor;
		}
		k++;
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

	template<ushort_t T, ushort_t U>
	using lambda_tt=BSplineCoeff<T, P, U>;

	using seq=typename make_gt_integer_sequence<ushort_t, Dim>::type;

	template <ushort_t ... Ids>
	using spline_tt= typename apply_gt_integer_sequence<seq>::template apply_tt<GenericBSpline, lambda_tt, Ids... >::type;

    public:

        /** @brief constructor */
        b_spline( ) :
            parametric_space<P, Dim>()
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
            return gt_pow<Dim>::apply(P);
        }


        template<uint_t F, uint_t S>
        struct lambda_get_first{
            static const uint_t value=F;
        };


        template <typename Quad, typename Storage, uint_t I, uint_t J>
        struct functor_get_vals{
	    using array_t=array<double, P+P+1>;

	private:
            Quad const& m_quad;
            Storage& m_storage;
	    array<array_t, Dim> const& m_knots;

	public:
            functor_get_vals(Quad const& quad_points_, Storage& storage_, array<array_t, Dim> const& knots_)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots(knots_)
	    {}

            template <typename Id>
            void operator()(Id){

                // using layout_t= typename apply_gt_integer_sequence
                //     <make_gt_itege_sequence
                //      <sizeof...(Dims...)>::apply_c_tt
                //      <layout_map, lambda_get_first, Dims..., Id::value>::type;

                // layout_t::fuck();
                //computing the strides at compile time
                // constexpr meta_storage_base<0,
                //                             // layout_t
                //                             layout_map<0,1,2>
                //                             > indexing{Dims...,Id::value};

                // defining the b-spline in I-J
                spline_tt<I, J, Id::value> basis_(m_knots);

		for (int k=0; k< m_quad.dimension(0); ++k)
		{
#ifdef VERBOSE
		    std::cout<<
			"evaluation of basis<"<<I<<", "<<J<<", "<<Id::value<<
			"> on point ("<<m_quad(k, 0)<<", "<<m_quad(k, 1)<<", "<<m_quad(k, 2)<<
			") gives: "<<basis_.evaluate(m_quad(k, 0),m_quad(k, 1),m_quad(k, 2))
					      <<std::endl;
#endif
		    // define a storage_metadata here! ==> generic dimension/layout
		     auto storage_index=(I-1)+P*(J-1)+P*P*(Id::value-1);
                    //accessign the data
		    m_storage(storage_index,k)=basis_.evaluate(m_quad(k, 0),m_quad(k, 1),m_quad(k, 2));
		}
            }
        };


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

		//unroll according to dimensions
		nest_loop
		    < array<array<double, P+P+1>, Dim>, Quad, Storage, functor_get_vals
		     , boost::mpl::range_c<uint_t, 1, P+1>
		     , boost::mpl::range_c<uint_t, 1, P+1>
		     , boost::mpl::range_c<uint_t, 1, P+1> >
		    (quad_points_, storage_, this->m_knots)();
		break;
	    default:
	    {
		std::cout<<"Operator not supported"<<std::endl;
		assert(false);
	    }
            }
        }
    };



}//namespace gridtools
