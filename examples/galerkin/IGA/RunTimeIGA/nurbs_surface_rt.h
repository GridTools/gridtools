#pragma once

#include <array>
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
		NurbsCurve(const std::array<double,N+P+1>& i_knots, const double* i_weights, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
					:PolinomialParametricCurve<DIM>(i_controlPoints)
					,m_weights(i_weights)
					,m_numeratorJacobianCurve(nullptr)
					,m_denominatorJacobianCurve(nullptr)
					,m_knots(i_knots)
					{}

		~NurbsCurve(void)
		{
			if(m_numeratorJacobianCurve!=nullptr)
			{
				delete m_numeratorJacobianCurve;
				delete m_denominatorJacobianCurve;
			}
		}


		/**
		 * @brief Curve jacobian evaluation method
		 * @param i_csi Parametric space point value
		 * @return Point in R^DIM
		 */
		Point<DIM> evaluateJacobian(double i_csi) const;


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


		// TODO: very bad design (see similar note for b-splines surface)
		// TODO: a similar data member is present also in b-spline surface class, better design should be available
		/**
		 * numerator b-spline curve jacobian (corresponding to a new b-spline curve)
		 */
		mutable BSplineCurve<P-1,N-1,DIM>* m_numeratorJacobianCurve;

		// TODO: very bad design (see similar note for b-splines surface)
		/**
		 * denominator b-spline curve jacobian (corresponding to a new b-spline curve)
		 */
		mutable BSplineCurve<P-1,N-1,1>* m_denominatorJacobianCurve;

		/**
		 * nurbs basis knots set // TODO: this data member is present also in b-splines curve, it should be in base class
		 */
		const std::array<double,N+P+1> m_knots;

	};

	template<int P, int N, int DIM>
	Point<DIM> NurbsCurve<P,N,DIM>::evaluateJacobian(const double i_csi) const
	{
		// Build numerator and denominator  b-spline derivatives (if needed)
		if(m_numeratorJacobianCurve == nullptr)
		{
			// TODO: almost all the code which follows is already implemented in b-spline curve jacobian calculation, refactoring required!

			// Build derivative knot set
			std::array<double,N+P-1> derivativeKnots;
			// TODO: switch to stl algos (and the next loop is useless due to initialization above)
			for(unsigned int i=0;i<P;++i)
			{
				derivativeKnots[i] = m_knots.front();
			}
			for(unsigned int i=0;i<P;++i)
			{
				derivativeKnots[N - 1 + i] = m_knots.back();
			}
			if(N-P-1>0)
			{
				std::copy(m_knots.begin()+P+1,
						  m_knots.begin()+N,
						  derivativeKnots.begin()+P);
			}

			// Build numerator derivative control node set
			std::vector<Point<DIM>> numeratorDerivativeControlPoints(N-1);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			for(unsigned int i=0;i<N-1;++i)
			{
				// TODO: the control point * weight multiplication is already performed in nurbs basis...
				numeratorDerivativeControlPoints[i] =
						P*(m_weights[i+1]*PolinomialParametricCurve<DIM>::m_controlPoints[i+1] - m_weights[i]*PolinomialParametricCurve<DIM>::m_controlPoints[i])/
						(m_knots[i+P+1] - m_knots[i+1]);
			}

			// Build denominator derivative control node set
			std::vector<Point<1>> denominatorDerivativeControlPoints(N-1);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			for(unsigned int i=0;i<N-1;++i)
			{
				denominatorDerivativeControlPoints[i].m_coords[0] = P*(m_weights[i+1] - m_weights[i])/(m_knots[i+P+1] - m_knots[i+1]);
			}

			// Build numerator and denominator b-splines curve derivatives
			m_numeratorJacobianCurve = new BSplineCurve<P-1,N-1,DIM>(derivativeKnots, numeratorDerivativeControlPoints);
			m_denominatorJacobianCurve = new BSplineCurve<P-1,N-1,1>(derivativeKnots, denominatorDerivativeControlPoints);

		}

		 Point<DIM> jacobian(m_numeratorJacobianCurve->evaluate(i_csi) - m_denominatorJacobianCurve->evaluate(i_csi).m_coords[0]*PolinomialParametricCurve<DIM>::evaluate(i_csi)); // TODO: check if externally the NURBS curve is already evaluated (evaluate(i_csi))

		 return dynamic_cast<NurbsBasis<P,N>*>(PolinomialParametricCurve<DIM>::m_basis)->getNormalizationFactor()*jacobian;
	}

	template<int P, int N, int DIM>
	PolinomialParametricBasis* NurbsCurve<P,N,DIM>::generateBasis(void) const
	{
		return new NurbsBasis<P,N>(m_knots,m_weights);
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
		NurbsSurface(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+P2+1>& i_knots2, const double* i_weights, const std::vector<Point<DIM> >& i_controlPoints)
					:PolinomialParametricSurface<DIM>(i_controlPoints)
					,m_weights(i_weights)
					,m_numeratorJacobianSurfaces{nullptr,nullptr}
					,m_denominatorJacobianSurfaces{nullptr,nullptr}
					,m_knots1(i_knots1)
					,m_knots2(i_knots2)
					{}

		/**
		 * @brief Surface jacobian evaluation method
		 * @param i_csi Parametric space point value
		 * @param i_eta Parametric space point value (second direction)
		 * @return Point in R^DIM (first and second parameter derivatives)
		 */
		std::array<Point<DIM>,2> evaluateJacobian(double i_csi, double i_eta) const;

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


		// TODO: very bad design (see similar note for b-splines surface)
		// TODO: a similar data member is present also in b-spline surface class, better design should be available
		/**
		 * numerator b-spline surfaces jacobian (corresponding to 2 new b-spline surfaces, one per derivation variable)
		 */
		mutable std::array<PolinomialParametricSurface<DIM>*,2> m_numeratorJacobianSurfaces; // TODO: a boost tuple could be used here


		// TODO: very bad design (see similar note for b-splines surface)
		/**
		 * denominator b-spline surfaces jacobian (corresponding to 2 new b-spline surfaces, one per derivation variable)
		 */
		mutable std::array<PolinomialParametricSurface<1>*,2> m_denominatorJacobianSurfaces;

		/**
		 * Nurbs basis knots set first direction// TODO: this data member is present also in b-splines surface, it should be in base class
		 */
		const std::array<double,N1+P1+1> m_knots1;

		/**
		 * Nurbs basis knots set second direction// TODO: this data member is present also in b-splines surface, it should be in base class
		 */
		const std::array<double,N2+P2+1> m_knots2;

	};


	template<int P1, int N1, int P2, int N2, int DIM>
	std::array<Point<DIM>,2> NurbsSurface<P1,N1,P2,N2,DIM>::evaluateJacobian(const double i_csi, const double i_eta) const
	{
		// Build numerator and denominator  b-spline derivatives (if needed)
		if(m_numeratorJacobianSurfaces[0] == nullptr)
		{
			// TODO: almost all the code which follows is already implemented in b-spline curve jacobian calculation, refactoring required!
			// TODO: refactoring available with curve jacobian case

			//////// First variable derivative elements calculation ////////

			// Build first variable derivative knot set
			std::array<double,N1+P1-1> derivativeKnots1;
			// TODO: switch to stl algos (and the next loop is useless due to initialization above)
			for(unsigned int i=0;i<P1;++i)
			{
				derivativeKnots1[i] = m_knots1.front();
			}
			for(unsigned int i=0;i<P1;++i)
			{
				derivativeKnots1[N1 - 1 + i] = m_knots1.back();
			}
			if(N1-P1-1>0)
			{
				std::copy(m_knots1.begin()+P1+1,
						  m_knots1.begin()+N1,
						  derivativeKnots1.begin()+P1);
			}

			// Build numerator derivative control node set
			std::vector<Point<DIM>> numeratorDerivativeControlPoints((N1-1)*N2);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			unsigned int global(0);
			for(unsigned int i=0;i<N1-1;++i)
			{
				for(unsigned int j=0;j<N2;++j,++global)
				{
					numeratorDerivativeControlPoints[global] =
							P1*(m_weights[N2*(i+1)+j]*PolinomialParametricSurface<DIM>::m_controlPoints[N2*(i+1)+j] - m_weights[N2*i+j]*PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots1[i+P1+1] - m_knots1[i+1]);
				}
			}

			// Build denominator derivative control node set
			std::vector<Point<1>> denominatorDerivativeControlPoints((N1-1)*N2);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			global = 0;
			for(unsigned int i=0;i<N1-1;++i)
			{
				for(unsigned int j=0;j<N2;++j,++global)
				{
					denominatorDerivativeControlPoints[global].m_coords[0] = P1*(m_weights[N2*(i+1)+j] - m_weights[N2*i+j])/(m_knots1[i+P1+1] - m_knots1[i+1]);
				}
			}

			// Build numerator and denominator b-splines surface first coordinate derivatives
			m_numeratorJacobianSurfaces[0] = new BSplineSurface<P1-1,N1-1,P2,N2,DIM>(derivativeKnots1,m_knots2,numeratorDerivativeControlPoints);
			m_denominatorJacobianSurfaces[0] = new BSplineSurface<P1-1,N1-1,P2,N2,1>(derivativeKnots1,m_knots2,denominatorDerivativeControlPoints);



			//////// Second variable derivative elements calculation ////////

			// Build first variable derivative knot set
			std::array<double,N2+P2-1> derivativeKnots2;
			// TODO: switch to stl algos (and the next loop is useless due to initialization above)
			for(unsigned int i=0;i<P2;++i)
			{
				derivativeKnots2[i] = m_knots2.front();
			}
			for(unsigned int i=0;i<P2;++i)
			{
				derivativeKnots2[N2 - 1 + i] = m_knots2.back();
			}
			if(N2-P2-1>0)
			{
				std::copy(m_knots2.begin()+P2+1,
						  m_knots2.begin()+N2,
						  derivativeKnots2.begin()+P2);
			}

			// Build numerator derivative control node set
			numeratorDerivativeControlPoints.resize(N1*(N2-1));
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			global = 0;
			for(unsigned int i=0;i<N1;++i)
			{
				for(unsigned int j=0;j<N2-1;++j,++global)
				{
					numeratorDerivativeControlPoints[global] =
							P2*(m_weights[N2*i+j+1]*PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j+1] - m_weights[N2*i+j]*PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots2[j+P2+1] - m_knots2[j+1]);
				}
			}

			// Build denominator derivative control node set
			denominatorDerivativeControlPoints.resize(N1*(N2-1));
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			global = 0;
			for(unsigned int i=0;i<N1;++i)
			{
				for(unsigned int j=0;j<N2-1;++j,++global)
				{
					denominatorDerivativeControlPoints[global].m_coords[0] = P2*(m_weights[N2*i+j+1] - m_weights[N2*i+j])/(m_knots2[j+P2+1] - m_knots2[j+1]);
				}
			}

			// Build numerator and denominator b-splines surface second coordinate derivatives
			m_numeratorJacobianSurfaces[1] = new BSplineSurface<P1,N1,P2-1,N2-1,DIM>(m_knots1,derivativeKnots2,numeratorDerivativeControlPoints);
			m_denominatorJacobianSurfaces[1] = new BSplineSurface<P1,N1,P2-1,N2-1,1>(m_knots1,derivativeKnots2,denominatorDerivativeControlPoints);

		}

		const Point<DIM> nurbsSurfaceValue(PolinomialParametricSurface<DIM>::evaluate(i_csi,i_eta));
		std::array<Point<DIM>,2> o_jacobian{m_numeratorJacobianSurfaces[0]->evaluate(i_csi,i_eta) - m_denominatorJacobianSurfaces[0]->evaluate(i_csi,i_eta).m_coords[0]*nurbsSurfaceValue,
										    m_numeratorJacobianSurfaces[1]->evaluate(i_csi,i_eta) - m_denominatorJacobianSurfaces[1]->evaluate(i_csi,i_eta).m_coords[0]*nurbsSurfaceValue};// TODO: check if externally the NURBS curve is already evaluated (evaluate(i_csi))


		const double normalizationFactor(dynamic_cast<BivariateNurbsBasis<P1,N1,P2,N2>*>(PolinomialParametricSurface<DIM>::m_basis)->getNormalizationFactor());

		o_jacobian[0] *= normalizationFactor;
		o_jacobian[1] *= normalizationFactor;

		return o_jacobian;
	}


	template<int P1, int N1, int P2, int N2, int DIM>
	BivariatePolinomialParametricBasis* NurbsSurface<P1,N1,P2,N2,DIM>::generateBasis(void) const
	{
		return new BivariateNurbsBasis<P1,N1,P2,N2>(m_knots1,m_knots2,m_weights);
	}

}
