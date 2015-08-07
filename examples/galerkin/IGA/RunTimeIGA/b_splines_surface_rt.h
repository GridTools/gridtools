#include <vector>
#include <numerics>
#include "point.h"
#include "b_splines_basis_rt.h"

#pragma once
namespace b_splines_rt
{
	// TODO: I am using this class for NURBS calculation, with Point<1> used as weight. Find better soultion
	/**
	 * @class b-spline curve calculation class
	 * @brief Class for the calculation of a b-spline curve value at a given point in parametric space given a set of knots and control points
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 * @tparam DIM co-domain number of dimensions
	 */
	template<int P, int N, int DIM>
	class BSplineCurve
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
					:m_controlPoints(i_controlPoints)
					,m_bsplineBasis(i_knots)
					{}

		/**
		 * @brief Curve evaluation method at a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Point in R^DIM
		 */
		Point<DIM> evaluate(double i_csi) const;


		/**
		 * @brief Weighted single basis function values at a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Weighted single basis function values in R^DIM
		 */
		std::vector<Point<DIM> > evaluate_weighted_basis(double i_csi) const;


	private:

		const std::vector<Point<DIM> >& m_controlPoints;// TODO: this can be avoided with the above modification (see contructor comment)

		const BSplineBasis<P,N> m_bsplineBasis;

	};

	template<int P, int N, int DIM>
	Point<DIM> BSplineCurve<P,N,DIM>::evaluate(const double i_csi) const
	{
		const std::vector<Point<DIM>> weighted_basis_values(evaluate_weighted_basis(i_csi));
		return std::accumulate(weighted_basis_values.begin(),weighted_basis_values.end(),Point<DIM>{});
	};


	template<int P, int N, int DIM>
	std::vector<Point<DIM>> BSplineCurve<P,N,DIM>::evaluate_weighted_basis(const double i_csi) const
	{
		std::vector<Point<DIM> > o_weighted_basis_values(m_bsplineBasis.evaluate(i_csi));

		// TODO: switch to std algos
		for(int i=0;i<N;++i)
		{
			o_weighted_basis_values[i] *= m_controlPoints[i];
		}

		return o_weighted_basis_values;
	};


	// TODO: I am using this class for NURBS calculation, with Point<1> used as weight. Find better soultion
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
	class BSplineSurface
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
				      :m_controlPoints(i_controlPoints)
					  ,m_bivariateBSplineBasis(i_knots1,i_knots2)
					  {}

		/**
		 * @brief Surface evaluation method at a given point in parametric space
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return Point in R^DIM
		 */
		Point<DIM> evaluate(double i_csi, double i_eta) const;


		/**
		 * @brief Weighted single basis function values at a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @param i_eta Parametric space point value (second direction)
		 * @return Weighted single basis function values in R^DIM
		 */
		std::vector<Point<DIM> > evaluate_weighted_basis(double i_csi, double i_eta) const;


	private:

		const std::vector<Point<DIM> >& m_controlPoints;

		const BivariateBSplineBasis<P1,N1,P2,N2> m_bivariateBSplineBasis;
	};

	template<int P1, int N1, int P2, int N2, int DIM>
	Point<DIM> BSplineSurface<P1,N1,P2,N2,DIM>::evaluate(const double i_csi, const double i_eta) const
	{
		const std::vector<Point<DIM>> weighted_basis_values(evaluate_weighted_basis(i_csi,i_eta));
		return std::accumulate(weighted_basis_values.begin(),weighted_basis_values.end(),Point<DIM>{});
	};

	template<int P1, int N1, int P2, int N2, int DIM>
	std::vector<Point<DIM>> BSplineSurface<P1,N1,P2,N2,DIM>::evaluate_weighted_basis(const double i_csi, const double i_eta) const
	{
		std::vector<Point<DIM> > o_weighted_basis_values(m_bivariateBSplineBasis.evaluate(i_csi,i_eta));

		// TODO: switch to std algos
		for(int i=0;i<N;++i)
		{
			o_weighted_basis_values[i] *= m_controlPoints[i];
		}

		return o_weighted_basis_values;
	};

}
