#pragma once

#include "generic_basis_rt.h"
#include "point.h"
#include <vector>

namespace iga_rt
{

	// TODO: change class names, they are just the interfaces for generic basis functions (or functions in R^N)

	/**
	 * @class Generic function basis curve calculation class
	 * @brief Class for the calculation of a generic function basis curve value at a given point in parametric space given a set of knots and control points
	 * @tparam DIM co-domain number of dimensions
	 */
	template<int DIM>
	class PolinomialParametricCurve
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots basis knot set
		 * @param i_controlPoints control point set in R^DIM
		 */
		PolinomialParametricCurve(const double* i_knots, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
								 :m_controlPoints(i_controlPoints)
								 ,m_basis(0)
								 ,m_knots(i_knots)
								 {}

		virtual ~PolinomialParametricCurve(void) { delete m_basis; }

		/**
		 * @brief Curve evaluation method at a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Point in R^DIM
		 */
		Point<DIM> evaluate(double i_csi) const;


	protected:

		/**
		 * Non copiable class
		 */
		PolinomialParametricCurve(const PolinomialParametricCurve<DIM>&);

		/**
		 * Non assignable class
		 */
		PolinomialParametricCurve<DIM>& operator=(const PolinomialParametricCurve<DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 */
		virtual PolinomialParametricBasis* generateBasis(void) const = 0;

		const std::vector<Point<DIM> >& m_controlPoints;// TODO: this can be avoided with the above modification (see constructor comment)

		mutable PolinomialParametricBasis* m_basis;// TODO: bad design, avoid mutable

		const double* m_knots;

	};

	template<int DIM>
	Point<DIM> PolinomialParametricCurve<DIM>::evaluate(const double i_csi) const
	{

		// TODO: find better solution to avoid this step
		if(m_basis == 0)
		{
			generateBasis();
		}

		std::vector<double> basis_values(m_basis->evaluate(i_csi));

		// TODO: switch to std algos
		Point<DIM> o_surface_value;
		for(int i=0;i<basis_values.size();++i)
		{
			o_surface_value += basis_values[i]*m_controlPoints[i];
		}

		return o_surface_value;
	};


	// TODO: change class names, they are just the interfaces for generic basis functions (or functions in R^N)
	/**
	 * @class Generic function basis surface calculation class
	 * @brief Class for the calculation of a generic function basis surface value at a given point in parametric space given a set of knots and control points
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int DIM>
	class PolinomialParametricSurface
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 polinomial basis knot set (first direction)
		 * @param i_knots2 polinomial basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		PolinomialParametricSurface(const double* i_knots1, const double* i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
								   :m_controlPoints(i_controlPoints)
								   ,m_basis(0)
								   ,m_knots1(i_knots1)
								   ,m_knots2(i_knots2)
								   {}

		virtual ~PolinomialParametricSurface(void) { delete m_basis; }

		/**
		 * @brief Surface evaluation method at a given point in parametric space
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return Point in R^DIM
		 */
		Point<DIM> evaluate(double i_csi, double i_eta) const;


	protected:

		/**
		 * Non copiable class
		 */
		PolinomialParametricSurface(const PolinomialParametricSurface<DIM>&);

		/**
		 * Non assignable class
		 */
		PolinomialParametricSurface<DIM>& operator=(const PolinomialParametricSurface<DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @param i_knots1 polinomial basis knot set (first direction)
		 * @param i_knots2 polinomial basis knot set (first direction)
		 */
		virtual BivariatePolinomialParametricBasis* generateBasis(void) const = 0;

		const std::vector<Point<DIM> >& m_controlPoints;

		mutable BivariatePolinomialParametricBasis* m_basis;// TODO: bad design, avoid mutable

		const double* m_knots1;

		const double* m_knots2;
	};

	template<int DIM>
	Point<DIM> PolinomialParametricSurface<DIM>::evaluate(const double i_csi, const double i_eta) const
	{
		if(m_basis==0)
		{
			m_basis = generateBasis();
		}

		std::vector<double> basis_values(m_basis->evaluate(i_csi,i_eta));

		// TODO: switch to std algos
		Point<DIM> o_surface_value;
		int global(0);
		for(int i=0;i<basis_values.size();++i)
		{
			o_surface_value += basis_values[i]*m_controlPoints[i];
		}


		return o_surface_value;
	};

}
