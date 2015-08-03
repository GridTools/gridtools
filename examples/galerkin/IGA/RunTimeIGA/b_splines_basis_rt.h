#pragma once
#include <vector>
#include "b_splines_rt.h"

namespace b_splines_rt
{

	/**
	 * @struct b-spline basis generation structure
	 * @brief Struct for recursive (compile time) generation of b-spline basis given a set of knots
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 */
	template<int P, int N>
	struct BSplineBasisGenerator
	{
		/**
		 * @brief b-spline generation method
		 * @param i_knots b-spline basis knot set
		 * @param io_bsplines vector of pointers to b-spline functions (ordered from 1 to N)
		 */
		static void generateBSplineBasis(const double* i_knots, std::vector<BaseBSpline*>& io_bsplines)
		{
			io_bsplines[N-1] = new BSpline<N,P>(i_knots);
			BSplineBasisGenerator<P,N-1>::generateBSplineBasis(i_knots, io_bsplines);
		}
	};

	template<int P>
	struct BSplineBasisGenerator<P,0>
	{
		static void generateBSplineBasis(const double* i_knots, std::vector<BaseBSpline*>& io_bsplines)
		{
		}
	};


	/**
	 * @class b-spline basis composition class
	 * @brief Class for the composition of b-spline basis given a set of knots
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 */
	template<int P, int N>
	class BSplineBasis
	{
	public:

		// TODO: use variadic template to generalize to different dimensions
		// TODO: add static assert to check values of template parameters

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 */
		BSplineBasis(const double* i_knots);

		virtual ~BSplineBasis(void);

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi) const;

	private:

		/**
		 * Non copyable class
		 */
		BSplineBasis(const BSplineBasis<P,N>&);

		/**
		 * Non assignable class
		 */
		BSplineBasis<P,N>& operator=(const BSplineBasis<P,N>&);

		/**
		 * b-spline base function set
		 */
		std::vector<BaseBSpline*> m_bsplines;
	};

	template<int P, int N>
	BSplineBasis<P,N>::BSplineBasis(const double* i_knots)
								   :m_bsplines(N,0)
	{
		BSplineBasisGenerator<P,N>::generateBSplineBasis(i_knots, m_bsplines);
	}

	template<int P, int N>
	BSplineBasis<P,N>::~BSplineBasis(void)
	{
		// TODO: use iterator
		for(int i=1;i<=N;++i)
		{
			delete m_bsplines[i-1];
		}
	}

	template<int P, int N>
	std::vector<double> BSplineBasis<P,N>::evaluate(const double i_csi) const
	{
		std::vector<double> o_bsplineValues(N);

		// TODO: use iterator
		for(int i=1;i<=N;++i)
		{
			o_bsplineValues[i-1] = m_bsplines[i-1]->evaluate(i_csi);
		}

		return o_bsplineValues;
	}


	/**
	 * @class Bivariate b-spline basis composition class
	 * @brief Class for the composition of bivariate b-spline basis given a set of knots
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order (second direction)
	 * @tparam N2 number of basis functions of order P2
	 */
	template<int P1, int N1, int P2, int N2>
	class BivariateBSplineBasis
	{
	public:

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (second direction)
		 */
		BivariateBSplineBasis(const double* i_knots1, const double* i_knots2)
							 :m_bsplineBasis1(i_knots1)
							 ,m_bsplineBasis2(i_knots2)
		{}

		virtual ~BivariateBSplineBasis(void)
		{}

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi, double i_eta) const;

	private:

		/**
		 * Non copyable class
		 */
		BivariateBSplineBasis(const BivariateBSplineBasis<P1,N1,P2,N2>&);

		/**
		 * Non assignable class
		 */
		BivariateBSplineBasis<P1,N1,P2,N2>& operator=(const BivariateBSplineBasis<P1,N1,P2,N2>&);

		/**
		 * B-spline basis in first direction
		 */
		BSplineBasis<P1,N1> m_bsplineBasis1;

		/**
		 * B-spline basis in second direction
		 */
		BSplineBasis<P2,N2> m_bsplineBasis2;
	};

	template<int P1, int N1, int P2, int N2>
	std::vector<double> BivariateBSplineBasis<P1,N1,P2,N2>::evaluate(const double i_csi, const double i_eta) const
	{
		std::vector<double> o_bivariateBSplineValues(N1*N2);

		const std::vector<double> bsplineValues1(m_bsplineBasis1.evaluate(i_csi));
		const std::vector<double> bsplineValues2(m_bsplineBasis1.evaluate(i_eta));

		int global(0);
		for(int i=0;i<N1;++i)
		{
			for(int j=0;j<N2;++j,++global)
			{
				o_bivariateBSplineValues[global] = bsplineValues1[i]*bsplineValues2[j];
			}
		}
		return o_bivariateBSplineValues;
	}

}
