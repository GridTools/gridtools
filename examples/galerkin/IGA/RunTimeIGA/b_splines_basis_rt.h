#pragma once
#include <vector>
#include <array>
#include "b_splines_rt.h"
#include "generic_basis_rt.h"

namespace iga_rt
{

	/**
	 * @struct b-spline basis generation structure
	 * @brief Struct for recursive (compile time) generation of b-spline basis given a set of knots
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 */
	template<int P, int N, int RUN_N=N>
	struct BSplineBasisGenerator
	{
		/**
		 * @brief b-spline generation method
		 * @param i_knots b-spline basis knot set
		 * @param io_bsplines vector of pointers to b-spline functions (ordered from 1 to N)
		 */
		static void generateBSplineBasis(const std::array<double,N+P+1>& i_knots, std::vector<BaseBSpline*>& io_bsplines)
		{
			io_bsplines[RUN_N-1] = new BSpline<RUN_N,P>(i_knots.data());
			BSplineBasisGenerator<P,N,RUN_N-1>::generateBSplineBasis(i_knots, io_bsplines);
		}
	};

	template<int P, int N>
	struct BSplineBasisGenerator<P,N,0>
	{
		static void generateBSplineBasis(const std::array<double,N+P+1>& i_knots, std::vector<BaseBSpline*>& io_bsplines)
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
	class BSplineBasis : public PolinomialParametricBasis
	{
	public:

		// TODO: use variadic template to generalize to different dimensions (or update other classes with constexpr, etc..)
		// TODO: add static assert to check values of template parameters
		// TODO: implement b_spline basis function evaluation selection with
		//		 node span method

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 */
		BSplineBasis(const std::array<double,N+P+1>& i_knots);

		virtual ~BSplineBasis(void);

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi) const;

		/**
		 * @brief Knot span finding method, returning the knot index i such as csi_{i}<=i_csi<csi_{i+1}
		 * @param i_csi Parametric space point value
		 * @return knot index respecting the condition described above
		 */
		unsigned int find_span(double i_csi) const;

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
		std::vector<BaseBSpline*> m_bsplines; // TODO: this should be const

		/**
		 * b-spline basis knot set number
		 */
		constexpr static unsigned int m_number_of_knots{N+P+1};

		/**
		 * b-spline basis knot set
		 */
		const std::array<double,m_number_of_knots> m_knots;

        /**
         * b-spline basis knot set last knot multiplicity status (required for correct calculation of basis value on knot set closure, multiplicity should be P+1 for interpolant basis)
         */
        const bool m_isLastKnotRepeated;

	};

	template<int P, int N>
	BSplineBasis<P,N>::BSplineBasis(const std::array<double,N+P+1>& i_knots)
								   :m_bsplines(N,0)
								   ,m_knots(i_knots)
                                   ,m_isLastKnotRepeated((i_knots[N+P]==i_knots[N]) ? true : false)// TODO: this check is based on the condition i_knots[i]<=i_knots[i+1] which is not checked anywhere
	{
		// TODO: add check on number of nodes
		// TODO: update interfaces also for other dimensions
		// TODO: do we have some ad-hoc container in grid tools for array+size?
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
		for(int i=1;i<N;++i)
		{
			o_bsplineValues[i-1] = m_bsplines[i-1]->evaluate(i_csi);
		}

		if(m_isLastKnotRepeated && i_csi == m_knots[N+P])
		{
			o_bsplineValues[N-1] = 1.;
		}
		else
		{
			o_bsplineValues[N-1] = m_bsplines[N-1]->evaluate(i_csi);
		}

		return o_bsplineValues;
	}

	template<int P, int N>
	unsigned int BSplineBasis<P,N>::find_span(const double i_csi) const
	{

		// TODO: switch to std algos
		// TODO: add check for knot ordering
		// TODO: add check for knot range: what if i_csi>m_knot(last) or i_csi<m_knot(first)

		if(i_csi == m_knots[0])
		{
			return 0;
		}

		for(unsigned int knot_index=1;knot_index<m_number_of_knots;++knot_index)
		{
			if(i_csi<m_knots[knot_index])
			{
				return knot_index-1;
			}
		}

		return N-1;
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
	class BivariateBSplineBasis : public BivariatePolinomialParametricBasis
	{
	public:

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (second direction)
		 */
		BivariateBSplineBasis(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+P2+1>& i_knots2);

		virtual ~BivariateBSplineBasis(void)
		{}

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi, double i_eta) const;

		/**
		 * @brief Knot span finding method, returning the knot index array {i} such as csi_{i}<=i_csi<csi_{i+1},  eta_{i}<=i_eta<eta_{i+1}
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return knot index array respecting the condition described above
		 */
		std::array<unsigned int,2> find_span(double i_csi, double i_eta) const;

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
		BSplineBasis<P1,N1> m_bsplines1;

		/**
		 * B-spline basis in second direction
		 */
		BSplineBasis<P2,N2> m_bsplines2;

		/**
		 * b-spline basis knot set number in first direction
		 */
		constexpr static unsigned int m_number_of_knots1{N1+P1+1};

		/**
		 * b-spline basis knot set number in second direction
		 */
		constexpr static unsigned int m_number_of_knots2{N2+P2+1};

	};

	template<int P1, int N1, int P2, int N2>
	BivariateBSplineBasis<P1,N1,P2,N2>::BivariateBSplineBasis(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+P2+1>& i_knots2)
															 :m_bsplines1(i_knots1)
															 ,m_bsplines2(i_knots2)
	{
	}


	template<int P1, int N1, int P2, int N2>
	std::vector<double> BivariateBSplineBasis<P1,N1,P2,N2>::evaluate(const double i_csi, const double i_eta) const
	{
		std::vector<double> o_bivariateBSplineValues(N1*N2);

		const std::vector<double> bsplineValues1(m_bsplines1.evaluate(i_csi));
		const std::vector<double> bsplineValues2(m_bsplines2.evaluate(i_eta));

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

	template<int P1, int N1, int P2, int N2>
	std::array<unsigned int,2> BivariateBSplineBasis<P1,N1,P2,N2>::find_span(const double i_csi, const double i_eta) const
	{

		// TODO: switch to std algos
		// TODO: add check for knot ordering
		// TODO: add check for knot range: what if i_csi>m_knot(last) or i_csi<m_knot(first)
		return std::array<unsigned int,2>{m_bsplines1.find_span(i_csi),m_bsplines2.find_span(i_eta)};
	}

}
