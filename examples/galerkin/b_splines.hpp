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

        template <typename ... Knots>
        GenericBSpline( Knots const& ... knots_ );

        /** @brief evaluates all the basis on the list of points*/
        template <typename ... Values>
        double evaluate(Values const& ... vals_) const{
            //check length vals_ == length m_univariate_bsplines
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
        //check that sizeof...(Knots) is same as Coeff
        m_univariate_bsplines(
            GenericBSpline<Coeff ...>::tuple_t(
                std::get<Coeff::dimension>(
                    std::make_tuple(&knots_ ...))...))
    {
    }

    template<ushort_t P, ushort_t Dim>
    struct parametric_space;

    template<ushort_t P>
    struct parametric_space<P, 3>{
        array<double, P+P+1> m_knots_i;
        array<double, P+P+1> m_knots_j;
        array<double, P+P+1> m_knots_k;
        //normalization

        parametric_space()
	{
	    int k=0;
            for(int i=0; i< (P+P+1)*2; i+=2)
            {
		std::cout<<" knots at "<<(double)((i)+1.-2.*P)<<std::endl;
                m_knots_i[k]=(double)((i)+1.-2.*P);///factor;
                m_knots_j[k]=(double)((i)+1.-2.*P);///factor;
                m_knots_k[k]=(double)((i)+1.-2.*P);///factor;
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
        //TODO use variadic
        template <ushort_t I, ushort_t J, ushort_t K>
        using spline_t = GenericBSpline<BSplineCoeff<0,P,I>, BSplineCoeff<1,P,J>, BSplineCoeff<2,P,K> >;
        //array<gridtools::array<double, P+1>, Dim> m_knots_i;
        //spline_t m_basis;

    public:

        /** @brief constructor */
        b_spline( ) :
            parametric_space<P, Dim>()
            // , m_basis(this->m_knots_i, this->m_knots_j, this->m_knots_k)
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


        template <typename Quad, typename Storage
		  , template<typename Q, typename S, uint_t ... I> class InnerFunctor
		  , typename Range1, typename Range2, typename Range3>
        struct nest_loop_IJK{

	    using array_t=array<double, P+P+1>;

            Quad const& m_quad;
            Storage& m_storage;
	    array_t const& m_knots_i, m_knots_j, m_knots_k;

            nest_loop_IJK(Quad const& quad_points_, Storage& storage_, array_t const& knots_i, array_t const& knots_j, array_t const& knots_k)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots_i(knots_i)
		, m_knots_j(knots_j)
		, m_knots_k(knots_k)
            {}

            void operator()(){
                boost::mpl::for_each<Range1>(nest_loop_J<Quad, Storage, InnerFunctor, Range2, Range3>(m_quad, m_storage, m_knots_i, m_knots_j, m_knots_k));
            }

        };


        template <typename Quad, typename Storage, template<typename Q, typename S, uint_t ... I> class InnerFunctor, typename Range2, typename Range3>
        struct nest_loop_J{

	    using array_t=array<double, P+P+1>;

            Quad const& m_quad;
            Storage& m_storage;
	    array_t const& m_knots_i, m_knots_j, m_knots_k;

            nest_loop_J(Quad const& quad_points_, Storage& storage_, array_t const& knots_i, array_t const& knots_j, array_t const& knots_k)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots_i(knots_i)
		, m_knots_j(knots_j)
		, m_knots_k(knots_k)
            {}

            template <typename Id>
            void operator()(Id ){
                boost::mpl::for_each<Range2>(nest_loop_K<Quad, Storage, InnerFunctor, Id::value, Range3>(m_quad, m_storage, m_knots_i, m_knots_j, m_knots_k));
            }

        };


        template <typename Quad, typename Storage, template<typename Q, typename S, uint_t ... I> class InnerFunctor, uint_t I, typename Range3>
        struct nest_loop_K{

	    using array_t=array<double, P+P+1>;

            Quad const& m_quad;
            Storage& m_storage;
	    array_t const& m_knots_i, m_knots_j, m_knots_k;

            nest_loop_K(Quad const& quad_points_, Storage& storage_, array_t const& knots_i, array_t const& knots_j, array_t const& knots_k)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots_i(knots_i)
		, m_knots_j(knots_j)
		, m_knots_k(knots_k)
            {}

            template <typename Id>
            void operator()(Id){
                boost::mpl::for_each<Range3>(InnerFunctor<Quad, Storage, I, Id::value>(m_quad, m_storage, m_knots_i, m_knots_j, m_knots_k));
            }

        };

        template <typename Quad, typename Storage, uint_t I, uint_t J>
        struct functor_get_vals{
	    using array_t=array<double, P+P+1>;

	private:
            Quad const& m_quad;
            Storage& m_storage;
	    array_t const& m_knots_i, m_knots_j, m_knots_k;

	public:
            functor_get_vals(Quad const& quad_points_, Storage& storage_, array_t const& knots_i, array_t const& knots_j, array_t const& knots_k)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots_i(knots_i)
		, m_knots_j(knots_j)
		, m_knots_k(knots_k)
	    {}

            template <typename Id>
            void operator()(Id){

                spline_t<I, J, Id::value> basis_(m_knots_i, m_knots_j, m_knots_k);

		for (int k=0; k< m_quad.dimension(0); ++k)
		{
		    std::cout<<
			"evaluation of basis<"<<I<<", "<<J<<", "<<Id::value<<
			"> on point ("<<m_quad(k, 0)<<", "<<m_quad(k, 1)<<", "<<m_quad(k, 2)<<
			") gives: "<<basis_.evaluate(m_quad(k, 0),m_quad(k, 1),m_quad(k, 2))
					      <<std::endl;
		    // define a storage_metadata here!
		    auto storage_index=(I-1)+P*(J-1)+P*P*(Id::value-1);
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

		nest_loop_IJK
		    <Quad, Storage, functor_get_vals
		     , boost::mpl::range_c<uint_t, 1, P+1>
		     , boost::mpl::range_c<uint_t, 1, P+1>
		     , boost::mpl::range_c<uint_t, 1, P+1> >
		    (quad_points_, storage_, this->m_knots_i, this->m_knots_j, this->m_knots_k)();
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
