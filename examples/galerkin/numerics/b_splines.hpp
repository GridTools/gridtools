#pragma once
#include "../IGA/RunTimeIGA/b_splines_rt.h"
#include <boost/fusion/include/accumulate.hpp>
#ifndef __CUDACC__
#include <tuple>
#endif
#include "nest_loops.hpp"
#include <common/generic_metafunctions/gt_integer_sequence.hpp>
#include <common/generic_metafunctions/gt_get.hpp>
#include <common/generic_metafunctions/accumulate.hpp>
#include "order.hpp"
// #include <storage/meta_storage_base.hpp>

namespace gdl{


    template<ushort_t Dim, typename Order, ushort_t I>
    struct BSplineCoeff;
    /**
       @brief struct containing ID and polynomial order
       \tparam Dim current dimension (for multivariate)
       \tparam I index
       \tparam P order
    */
    template<ushort_t Dim, ushort_t I, ushort_t ... P>
    struct BSplineCoeff<Dim, order<P...>, I>{
        static const ushort_t dimension=Dim;
        static const constexpr gt::array<ushort_t, sizeof...(P)> order{P...};
        static const ushort_t index=I;
    };

    /** @brief just to ease the notation*/
    template <typename Coeff>
    struct BSplineDerived : iga_rt::BSpline<Coeff::index, Coeff::order[Coeff::dimension]>
    {
        static const int index=Coeff::index;
        static const int dimension=Coeff::dimension;
        using super=iga_rt::BSpline<Coeff::index,Coeff::order[Coeff::dimension]>;
        //inheriting constructors
        using iga_rt::BSpline<Coeff::index,Coeff::order[Coeff::dimension]>::BSpline;
    };

    /**
       @biref functor used to get the values in a point of a multivatiate B-spline

       say that we have a point in 4D (x1,x2,x3,x4) and a tuple of 4 univariate B-spline
       (sp1, sp2, sp3, sp4), then one can use this functor together with (e.g.)
       boost::fusion::transform to obtain the vector (sp1(x1),sp2(x2),sp3(x3),sp4(x4) ).
     */
    template <typename Tuple>
    struct get_val
    {
    private:
        Tuple const& m_vals;
    public:
        get_val(Tuple const& vals_):
            m_vals(vals_)
        {}

        /**
           @brief evaluation operator

           note that Basis is a univariate B-spline in a specific dimension
         */
        template<typename Basis>
        double operator()(Basis const& basis_) const
        {
            return basis_.evaluate(std::get<Basis::dimension>(m_vals));
        }

        template<typename T>
        struct result{
        using type = double;
        };

    };

    /**
       @brief generic multivariate b-spline

       This class contains a tuple of univariate B-splines
    */
    template<typename ... Coeff>
    class GenericBSpline
    {
    private:
        using tuple_t=boost::fusion::vector<BSplineDerived<Coeff> ... >;
        tuple_t m_univariate_bsplines;

        struct multiplies_f{

            template<typename ... Args>
            double operator()(Args const& ... args_){
                return gt::multiplies()(args_ ...);
            }

            template<typename T>
            struct result{
                using type = double;
            };

        };
    public:

        /**
           @brief Constructor given a multidimensional array of knots

           @param knots_ an array of array whose dimension depends on the parametric space dimension
         */
        template <typename Knots>
        GenericBSpline( Knots const& knots_ );

        /** @brief evaluates all the basis on a list of points

            @tparam vals_ variadic pack (of length equal to the parametric space dimension) containing the point in
            which the B-spline is evaluated.
         */
        template <typename ... Values>
        double evaluate(Values const& ... vals_) const{

            GRIDTOOLS_STATIC_ASSERT((sizeof...(Values) == sizeof...(Coeff)), "mismatch between the B-splines dimension and the parametric space dimension");
            auto tuple_of_vals = boost::fusion::transform (
                m_univariate_bsplines
                , get_val<std::tuple<Values...> >(std::make_tuple(vals_...) ) );

            //initial state
            double state(1.);

            return boost::fusion::accumulate(
                //calls the evaluate, and returns a sequence of evaluation
                tuple_of_vals
                , state
                , multiplies_f()
                );
        }


    };

    // //old implementation
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
                &std::get<Coeff::dimension>(knots_)[0]...))
    {
    }

    template<typename Order>
    struct parametric_space;

    /**
       @brief class holding the parametric space

       The parametric space is depending only on the order of the basis,
       it will be mapped to the deformed configuration by the finite
       element transformation (thus depending on the position of the
       knots on the deformed grid)
     */
    template<ushort_t ... P>
    struct parametric_space<order<P ... > >{
        typedef std::tuple<gt::array<double, P+P+1>...> knots_t;
        static const ushort_t dim = (sizeof...(P));
        typedef boost::mpl::vector<static_ushort<P>...> orders_t;
        knots_t m_knots;

        struct assign_knots{

            knots_t & m_knots;
            assign_knots(knots_t& knots_): m_knots(knots_){}

            template <typename Id>
            void operator()(Id){
                //knots span from -2P+1 to 2P+1 with step of 2:
                //[-2P+1, -2P+3, ..., -1, 1, ..., 2P-1, 2P+1]
                //for P=2 : [-3, -1, 1, 3, 5]
                //TODO loop over d dimensions
                //using index_tuple=typename boost::mpl::size<Id>::type;

                const int vec_size_ = std::get<Id::value>(m_knots).size();
                int k=0;
                for(int i=0; i< (vec_size_)*2; i+=2)
                {
#ifdef VERBOSE
                    std::cout<<" knots at "<<(double)((i)-vec_size_+2)<<std::endl;
#endif
                    std::get<Id::value>(m_knots)[k]=(double)((i)-vec_size_+2);
                    k++;
                }

            }
        };

        parametric_space()
	{
            boost::mpl::for_each<boost::mpl::range_c< ushort_t, 0, sizeof...(P) > >(assign_knots(m_knots));
	}
    };


    template<typename Order>
    struct b_spline;
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
    template<ushort_t ... P>
    struct b_spline<order<P...> > : parametric_space<order<P...> > {

    private:

        static const ushort_t Dim=sizeof...(P);
	template<ushort_t T, ushort_t U>
	using lambda_tt=BSplineCoeff<T, order<P...>, U>;

	using seq=typename gt::make_gt_integer_sequence<ushort_t, Dim>::type;

	template <ushort_t ... Ids>
	using spline_tt= typename gt::apply_gt_integer_sequence<seq>::template apply_tt<GenericBSpline, lambda_tt, Ids... >::type;

    public:

        /** @brief constructor */
        b_spline( ) :
            parametric_space<order<P...> >()
        {
        }

        template <typename Storage>
        void getDofCoords(Storage& /*s*/){
            // should fill the input storage with the knots local coordinates
            // to be implemented
            assert(false);
        }

        /** compile-time known*/
        constexpr int getCardinality() const
        {
            //returns the number of basis functions (P)^dim
            return gt::accumulate(gt::multiplies(), P...);
        }


        /**@brief metafunction to get the first of two template arguments*/
        template<int F, int S>
        struct lambda_get_first{
            static const int value=F;
        };


        /**
           @brief functor to evaluate the B-spline basis on the quadrature points.

           It is plugged into a hierarchical loop structure.
           NOTE: this functor is not generic, is
         */
        template <typename ArrayKnots, typename Quad, typename Storage, int ... Dims // I, uint_t J
                  >
        struct functor_get_vals{
	    using array_t=ArrayKnots;

	private:
            Quad const& m_quad;
            Storage& m_storage;
	    ArrayKnots const& m_knots;

	public:
            functor_get_vals(Quad const& quad_points_, Storage& storage_, ArrayKnots const& knots_)
                :
                m_quad(quad_points_)
                , m_storage(storage_)
		, m_knots(knots_)
	    {}

            template<typename Id, typename Basis, typename T>
            struct functor_assign_storage;

            template<typename Id,typename Basis, ushort_t ... Integers>
            struct functor_assign_storage<Id, Basis, gt::gt_integer_sequence<ushort_t, Integers...> >{

                static void apply( Storage & storage_, Basis const& basis_, Quad const& quad_, int const& k)
                {
                    storage_(Id::value-1,k)=basis_.evaluate(quad_(k, Integers)...);
                }

            };

#ifdef __CUDACC__
            template<int_t PP, int_t NN>
            struct get_second{
                typedef static_int<NN> type;
                constexpr static const int_t value=NN;
                constexpr static const static_int<NN> value_=static_int<NN>();
            };
#endif

            template <typename Id>
            void operator()(Id){

                //innermost loop has lower stride <0,1,2,3,....>
                using layout_t= typename gt::apply_gt_integer_sequence
                    <typename gt::make_gt_integer_sequence
                     <int, sizeof...(Dims)+1>::type >::template apply_c_tt
                    <gt::layout_map, lambda_get_first, Dims..., Id::value>::type;

                // computing the strides at compile time
                // NOTE: __COUNTER__ is a non standard non very portable solution
                // though the main compilers implement it
                //TODO generalize

#ifdef __CUDACC__ // nvcc crap (quite amazing)
                static const constexpr gt::meta_storage_base<static_int<__COUNTER__>,layout_t,false> indexing{get_second<P,2>::value_ ... };
#else
                constexpr gt::meta_storage_base<static_int<__COUNTER__>,layout_t,false> indexing{P ...};
#endif
                using basis_t = spline_tt<Dims ... , Id::value>;
                basis_t basis_(m_knots);

		for (int k=0; k< m_quad.dimension(0); ++k)
		{
                    // std::cout<<
                    //     "evaluation of basis< "<<Id::value<<
                    //     "> on point ("<<m_quad(k, 0) <<", "<<m_quad(k, 1)//<<", "<<m_quad(k, 2)
                    //                            <<
                    //     ") gives: "<<basis_.evaluate(m_quad(k, 0), m_quad(k, 1))
                    //                            <<std::endl;

                    // array<int, sizeof...(Dims)+1> vals_{Dims..., Id::value};

                    functor_assign_storage<
                        //static_int<indexing.index(static_int<Dims-1>() ... , static_int<Id::value>())>
                        static_int<indexing.index(Dims-1 ... , Id::value)>
                        , basis_t
                        , typename gt::make_gt_integer_sequence<ushort_t, Dim>::type >
                        ::apply( m_storage, basis_, m_quad, k);

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

                GRIDTOOLS_STATIC_ASSERT((sizeof...(P)==Dim), "error");
		//unroll according to dimensions
                nest_loop
                    < std::tuple<gt::array<double, P+P+1>...>, Quad, Storage, functor_get_vals
                      , boost::mpl::range_c<int, 1, P+1> ...
                      >
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
