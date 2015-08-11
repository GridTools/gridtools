#pragma once

#include "point.h"
#include "nurbs_basis_rt.h"
#include "generic_basis_parametric_surface_rt.h"

namespace iga_rt{

	/**
	 * @class Nurbs curve calculation class
	 * @brief Class for the calculation of a nurbs curve value at a given point in parametric space given a set of knots, weights and control points
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 * @tparam DIM co-domain number of dimensions
	 */
	template<int P, int N, int DIM>
	class NurbsCurve : public PolinomialParametricCurve<DIM>
	{
	public:

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 * @param i_weights b-spline basis function linear combination weight set
		 * @param i_controlPoints control point set in R^DIM
		 */
		NurbsCurve(const double* i_knots, const double* i_weights, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
					:PolinomialParametricCurve<DIM>(i_knots,i_controlPoints)
					,m_weights(i_weights)
					{}

	private:

		/**
		 * Non copyable class
		 */
		NurbsCurve(NurbsCurve<P,N,DIM>&);

		/**
		 * Non assignable class
		 */
		NurbsCurve<P,N,DIM>& operator=(NurbsCurve<P,N,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @param i_knots basis knot set
		 * @param i_weights b-spline basis function linear combination weight set
		 * @return (new) pointer to basis
		 */
		PolinomialParametricBasis* generateBasis(void) const;

		const double* m_weights;

	};

	template<int P, int N, int DIM>
	PolinomialParametricBasis* NurbsCurve<P,N,DIM>::generateBasis(void) const
	{
		new NurbsBasis<P,N>(PolinomialParametricCurve<DIM>::m_knots,m_weights);
	}



	/**
	 * @class nurbs surface calculation class
	 * @brief Class for the calculation of a nurbs surface value at a given point in parametric space given a set of knots, weights and control points
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order (second direction)
	 * @tparam N1 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int P1, int N1, int P2, int N2, int DIM>
	class NurbsSurface : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, controlo points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_weights b-spline basis function linear combination weight set
		 * @param i_controlPoints control point set in R^DIM
		 */
		NurbsSurface(const double* i_knots1, const double* i_knots2, const double* i_weights, const std::vector<Point<DIM> >& i_controlPoints)
					:PolinomialParametricSurface<DIM>(i_knots1,i_knots2,i_controlPoints)
					,m_weights(i_weights)
					{}

	private:

		/**
		 * Non copiable class
		 */
		NurbsSurface(const NurbsSurface<P1,N1,P2,N2,DIM>&);

		/**
		 * Non assignable class
		 */
		NurbsSurface<P1,N1,P2,N2,DIM>& operator=(const NurbsSurface<P1,N1,P2,N2,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_weights b-spline basis function linear combination weight set
		 * @return (new) pointer to basis
		 */
		BivariatePolinomialParametricBasis* generateBasis(void) const;

		const double* m_weights;

	};

	template<int P1, int N1, int P2, int N2, int DIM>
	BivariatePolinomialParametricBasis* NurbsSurface<P1,N1,P2,N2,DIM>::generateBasis(void) const
	{
		new BivariateNurbsBasis<P1,N1,P2,N2>(PolinomialParametricSurface<DIM>::m_knots1,PolinomialParametricSurface<DIM>::m_knots2,m_weights);
	}

}
