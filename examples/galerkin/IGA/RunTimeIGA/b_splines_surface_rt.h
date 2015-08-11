#pragma once

#include "point.h"
#include "b_splines_basis_rt.h"
#include "generic_basis_parametric_surface_rt.h"

namespace iga_rt
{
	/**
	 * @class b-spline curve calculation class
	 * @brief Class for the calculation of a b-spline curve value at a given point in parametric space given a set of knots and control points
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 * @tparam DIM co-domain number of dimensions
	 */
	template<int P, int N, int DIM>
	class BSplineCurve : public PolinomialParametricCurve<DIM>
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineCurve(const double* i_knots, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
					:PolinomialParametricCurve<DIM>(i_knots,i_controlPoints)
					{}

	private:

		/**
		 * Non copyable class
		 */
		BSplineCurve(BSplineCurve<P,N,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineCurve<P,N,DIM>& operator=(BSplineCurve<P,N,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		PolinomialParametricBasis* generateBasis(void) const;

	};

	template<int P, int N, int DIM>
	PolinomialParametricBasis* BSplineCurve<P,N,DIM>::generateBasis(void) const
	{
		new BSplineBasis<P,N>(PolinomialParametricCurve<DIM>::m_knots);
	}



	// TODO:NURBS basis definition is similar to surface calculation with Point<1> used as weight. Find solution not using code duplication!
	/**
	 * @class b-spline surface calculation class
	 * @brief Class for the calculation of a b-spline surface value at a given point in parametric space given a set of knots and control points
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order (second direction)
	 * @tparam N1 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int P1, int N1, int P2, int N2, int DIM>
	class BSplineSurface : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, controlo points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineSurface(const double* i_knots1, const double* i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
				      :PolinomialParametricSurface<DIM>(i_knots1,i_knots2,i_controlPoints)
					  {}

	private:

		/**
		 * Non copiable class
		 */
		BSplineSurface(const BSplineSurface<P1,N1,P2,N2,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineSurface<P1,N1,P2,N2,DIM>& operator=(const BSplineSurface<P1,N1,P2,N2,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		BivariatePolinomialParametricBasis* generateBasis(void) const;

	};

	template<int P1, int N1, int P2, int N2, int DIM>
	BivariatePolinomialParametricBasis* BSplineSurface<P1,N1,P2,N2,DIM>::generateBasis(void) const
	{
		new BivariateBSplineBasis<P1,N1,P2,N2>(PolinomialParametricSurface<DIM>::m_knots1,PolinomialParametricSurface<DIM>::m_knots2);
	}
}
