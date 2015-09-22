#pragma once

#include "point.h"
#include "b_splines_basis_rt.h"
#include "generic_basis_parametric_surface_rt.h"
#include <array>
#include <algorithm>

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
		BSplineCurve(const std::array<double,N+P+1>& i_knots, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
					:PolinomialParametricCurve<DIM>(i_controlPoints)
					,m_jacobianCurve(nullptr)
					,m_knots(i_knots)
					{}

		~BSplineCurve(void)
		{
			if(m_jacobianCurve!=nullptr)
			{
				delete m_jacobianCurve;
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

		// TODO: very bad design, I would need a "GenericCurve/Function" which can be build as composition
		// of other GenericCurve/Function, as in the case of the Nurbs jacobian which is composed by 2
		// b-splines curves of different order/basis size and a scalar
		/**
		 * b-spline curve jacobian (corresponding to a new b-spline curve)
		 */
		mutable BSplineCurve<P-1,N-1,DIM>* m_jacobianCurve;

		/**
		 * b-splines basis knots set
		 */
		const std::array<double,N+P+1> m_knots;

	};

	template<int P, int N, int DIM>
	Point<DIM> BSplineCurve<P,N,DIM>::evaluateJacobian(const double i_csi) const
	{
		// Build jacobian b-spline curve (if needed)
		if(m_jacobianCurve == nullptr)
		{
			// Build derivative knot set
			std::array<double,N+P-1> derivativeKnots;
			// TODO: switch to stl algos (and the next loop is useless due to initialization above)
			for(unsigned int i=0;i<P;++i)
			{
				derivativeKnots[i] = m_knots.front();
			}
			for(unsigned int i=0;i<P;++i)
			{
				// TODO: avoid continuous access to m_knots
				derivativeKnots[N - 1 + i] = m_knots.back();
			}
			if(N-P-1>0)
			{
				std::copy(m_knots.begin()+P+1,
						  m_knots.begin()+N,
						  derivativeKnots.begin()+P);
			}

			// Build derivative control node set
			std::vector<Point<DIM> > derivativeControlPoints(N-1);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			for(unsigned int i=0;i<N-1;++i)
			{
				derivativeControlPoints[i] =
						P*(PolinomialParametricCurve<DIM>::m_controlPoints[i+1] - PolinomialParametricCurve<DIM>::m_controlPoints[i])/
						(m_knots[i+P+1] - m_knots[i+1]);
			}

			m_jacobianCurve = new BSplineCurve<P-1,N-1,DIM>(derivativeKnots, derivativeControlPoints);
		}

		return m_jacobianCurve->evaluate(i_csi);
	}

	template<int P, int N, int DIM>
	PolinomialParametricBasis* BSplineCurve<P,N,DIM>::generateBasis(void) const
	{
		return new BSplineBasis<P,N>(m_knots);
	}


	/**
	 * @class b-spline curve calculation class, degree partial specialization // TODO: as for the surface case this is required for jacobian calculation, check if better solution is possible
	 * @brief Class for the calculation of a b-spline curve value at a given point in parametric space given a set of knots and control points
	 * @tparam N number of basis functions of order P=0
	 * @tparam DIM co-domain number of dimensions
	 */
	template<int N, int DIM>
	class BSplineCurve<0,N,DIM> : public PolinomialParametricCurve<DIM>
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
		BSplineCurve(const std::array<double,N+1>& i_knots, const std::vector<Point<DIM> >& i_controlPoints)// TODO: change constructor signature and add setControlPoint methods (same for surface and volumes)
					:PolinomialParametricCurve<DIM>(i_controlPoints)
					,m_knots(i_knots)
					{}

		/**
		 * @brief Curve jacobian evaluation method
		 * @param i_csi Parametric space point value
		 * @return Point in R^DIM
		 */
		inline Point<DIM> evaluateJacobian(double i_csi) const { return Point<DIM>{}; }

	private:

		/**
		 * Non copyable class
		 */
		BSplineCurve(BSplineCurve<0,N,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineCurve<0,N,DIM>& operator=(BSplineCurve<0,N,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		PolinomialParametricBasis* generateBasis(void) const;

		/**
		 * b-splines basis knots set
		 */
		const std::array<double,N+1> m_knots;

	};

	template<int N, int DIM>
	PolinomialParametricBasis* BSplineCurve<0,N,DIM>::generateBasis(void) const
	{
		return new BSplineBasis<0,N>(m_knots);
	}


	// TODO:NURBS basis definition is similar to surface calculation with Point<1> used as weight. Find solution not using code duplication!
	/**
	 * @class b-spline surface calculation class
	 * @brief Class for the calculation of a b-spline surface value at a given point in parametric space given a set of knots and control points
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order (second direction)
	 * @tparam N2 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int P1, int N1, int P2, int N2, int DIM>
	class BSplineSurface : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineSurface(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+P2+1> i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
				      :PolinomialParametricSurface<DIM>(i_controlPoints)
					  ,m_jacobianSurfaces{nullptr,nullptr}
					  ,m_knots1(i_knots1)
					  ,m_knots2(i_knots2)
					  {}

		virtual ~BSplineSurface(void)
		{
			if(m_jacobianSurfaces[0] != nullptr)
			{
				delete m_jacobianSurfaces[0];
				delete m_jacobianSurfaces[1];
			}
		}

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

		// TODO: very bad design, I would need a "GenericCurve/Function" which can be build as composition
		// of other GenericCurve/Function, as in the case of the Nurbs jacobian which is composed by 2
		// b-splines curves of different order/basis size and a scalar
		/**
		 * b-spline surface jacobian (corresponding to 2 new b-spline surfaces, one per derivation variable)
		 */
		mutable std::array<PolinomialParametricSurface<DIM>*,2> m_jacobianSurfaces; // TODO: a boost tuple could be used here

		/**
		 * b-splines basis knots set first direction
		 */
		const std::array<double,N1+P1+1> m_knots1;

		/**
		 * b-splines basis knots set second direction
		 */
		const std::array<double,N2+P2+1> m_knots2;

	};

	template<int P1, int N1, int P2, int N2, int DIM>
	std::array<Point<DIM>,2> BSplineSurface<P1,N1,P2,N2,DIM>::evaluateJacobian(const double i_csi,const double i_eta) const
	{

		if(m_jacobianSurfaces[0] == nullptr)
		{
			// TODO: refactoring wrt 1D case

			// Build first derivative knot set
			std::array<double,N1+P1-1> derivativeKnots1;
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


			// Build derivative control point set
			std::vector<Point<DIM>> derivativeControlPoints((N1-1)*N2);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			unsigned int global(0);
			for(unsigned int i=0;i<N1-1;++i)
			{
				for(unsigned int j=0;j<N2;++j,++global)
				{
					derivativeControlPoints[global] =
							P1*(PolinomialParametricSurface<DIM>::m_controlPoints[N2*(i+1)+j] - PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots1[i+P1+1] - m_knots1[i+1]);
				}
			}

			m_jacobianSurfaces[0] = new BSplineSurface<P1-1,N1-1,P2,N2,DIM>(derivativeKnots1,m_knots2,derivativeControlPoints);


			// Build second derivative knot set
			std::array<double,N2+P2-1> derivativeKnots2;
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

			// Build second derivative control point set
			derivativeControlPoints.resize(N1*(N2-1));
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			global = 0;
			for(unsigned int i=0;i<N1;++i)
			{
				for(unsigned int j=0;j<N2-1;++j,++global)
				{
					derivativeControlPoints[global] =
							P2*(PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j+1] - PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots2[j+P2+1] - m_knots2[j+1]);
				}
			}

			m_jacobianSurfaces[1] = new BSplineSurface<P1,N1,P2-1,N2-1,DIM>(m_knots1,derivativeKnots2,derivativeControlPoints);
		}

		return std::array<Point<DIM>,2>{m_jacobianSurfaces[0]->evaluate(i_csi,i_eta),m_jacobianSurfaces[1]->evaluate(i_csi,i_eta)};

	}

	template<int P1, int N1, int P2, int N2, int DIM>
	BivariatePolinomialParametricBasis* BSplineSurface<P1,N1,P2,N2,DIM>::generateBasis(void) const
	{
		return new BivariateBSplineBasis<P1,N1,P2,N2>(m_knots1,m_knots2);
	}

	/**
	 * @class b-spline surface calculation class, first direction degree partial specialization // TODO: check if better solution is possible
	 * @brief Class for the calculation of a b-spline surface value at a given point in parametric space given a set of knots and control points
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order (second direction)
	 * @tparam N2 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int N1, int P2, int N2, int DIM>
	class BSplineSurface<0,N1,P2,N2,DIM> : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineSurface(const std::array<double,N1+1>& i_knots1, const std::array<double,N2+P2+1> i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
					  :PolinomialParametricSurface<DIM>(i_controlPoints)
					  ,m_jacobianSurface{nullptr}
					  ,m_knots1(i_knots1)
					  ,m_knots2(i_knots2)
					  {}

		virtual ~BSplineSurface(void)
		{
			if(m_jacobianSurface != nullptr)
			{
				delete m_jacobianSurface;
			}
		}

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
		BSplineSurface(const BSplineSurface<0,N1,P2,N2,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineSurface<0,N1,P2,N2,DIM>& operator=(const BSplineSurface<0,N1,P2,N2,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		BivariatePolinomialParametricBasis* generateBasis(void) const;

		/**
		 * b-spline surface jacobian (corresponding to a new b-spline surfaces, for second direction variable derivation)
		 */
		mutable PolinomialParametricSurface<DIM>* m_jacobianSurface;

		/**
		 * b-splines basis knots set first direction
		 */
		const std::array<double,N1+1> m_knots1;

		/**
		 * b-splines basis knots set second direction
		 */
		const std::array<double,N2+P2+1> m_knots2;

	};


	template<int N1, int P2, int N2, int DIM>
	std::array<Point<DIM>,2> BSplineSurface<0,N1,P2,N2,DIM>::evaluateJacobian(const double i_csi,const double i_eta) const
	{
		if(m_jacobianSurface == nullptr)
		{
			// TODO: refactoring wrt 1D case
			// TODO: refactoring with generic templated case

			// Build second derivative knot set
			std::array<double,N2+P2-1> derivativeKnots2;
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

			// Build second derivative control point set
			std::vector<Point<DIM>> derivativeControlPoints(N1*(N2-1));
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			unsigned int global(0);
			for(unsigned int i=0;i<N1;++i)
			{
				for(unsigned int j=0;j<N2-1;++j,++global)
				{
					derivativeControlPoints[global] =
							P2*(PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j+1] - PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots2[j+P2+1] - m_knots2[j+1]);
				}
			}

			m_jacobianSurface = new BSplineSurface<0,N1,P2-1,N2-1,DIM>(m_knots1,derivativeKnots2,derivativeControlPoints);
		}

		return std::array<Point<DIM>,2>{Point<DIM>{},m_jacobianSurface->evaluate(i_csi,i_eta)};
	}

	template<int N1, int P2, int N2, int DIM>
	BivariatePolinomialParametricBasis* BSplineSurface<0,N1,P2,N2,DIM>::generateBasis(void) const
	{
		return new BivariateBSplineBasis<0,N1,P2,N2>(m_knots1,m_knots2);
	}


	/**
	 * @class b-spline surface calculation class, second direction degree partial specialization // TODO: check if better solution is possible
	 * @brief Class for the calculation of a b-spline surface value at a given point in parametric space given a set of knots and control points
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam N2 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int P1, int N1, int N2, int DIM>
	class BSplineSurface<P1,N1,0,N2,DIM> : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineSurface(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+1> i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
					  :PolinomialParametricSurface<DIM>(i_controlPoints)
					  ,m_jacobianSurface{nullptr}
					  ,m_knots1(i_knots1)
					  ,m_knots2(i_knots2)
					  {}

		virtual ~BSplineSurface(void)
		{
			if(m_jacobianSurface != nullptr)
			{
				delete m_jacobianSurface;
			}
		}

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
		BSplineSurface(const BSplineSurface<P1,N1,0,N2,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineSurface<P1,N1,0,N2,DIM>& operator=(const BSplineSurface<P1,N1,0,N2,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		BivariatePolinomialParametricBasis* generateBasis(void) const;

		/**
		 * b-spline surface jacobian (corresponding to a new b-spline surfaces, for first direction variable derivation)
		 */
		mutable PolinomialParametricSurface<DIM>* m_jacobianSurface;

		/**
		 * b-splines basis knots set first direction
		 */
		const std::array<double,N1+P1+1> m_knots1;

		/**
		 * b-splines basis knots set second direction
		 */
		const std::array<double,N2+1> m_knots2;

	};

	template<int P1, int N1, int N2, int DIM>
	std::array<Point<DIM>,2> BSplineSurface<P1,N1,0,N2,DIM>::evaluateJacobian(const double i_csi,const double i_eta) const
	{
		if(m_jacobianSurface == nullptr)
		{
			// TODO: refactoring wrt 1D case
			// TODO: refactoring with generic templated case

			// Build first derivative knot set
			std::array<double,N1+P1-1> derivativeKnots1;
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


			// Build derivative control point set
			std::vector<Point<DIM>> derivativeControlPoints((N1-1)*N2);
			// TODO: use std::algos
			// TODO: what about these loops (loop over control points and basis functions), should I use GridTools here?
			unsigned int global(0);
			for(unsigned int i=0;i<N1-1;++i)
			{
				for(unsigned int j=0;j<N2;++j,++global)
				{
					derivativeControlPoints[global] =
							P1*(PolinomialParametricSurface<DIM>::m_controlPoints[N2*(i+1)+j] - PolinomialParametricSurface<DIM>::m_controlPoints[N2*i+j])/
							(m_knots1[i+P1+1] - m_knots1[i+1]);
				}
			}

			m_jacobianSurface = new BSplineSurface<P1-1,N1-1,0,N2,DIM>(derivativeKnots1,m_knots2,derivativeControlPoints);

		}

		return std::array<Point<DIM>,2>{m_jacobianSurface->evaluate(i_csi,i_eta),Point<DIM>{}};
	}

	template<int P1, int N1, int N2, int DIM>
	BivariatePolinomialParametricBasis* BSplineSurface<P1,N1,0,N2,DIM>::generateBasis(void) const
	{
		return new BivariateBSplineBasis<P1,N1,0,N2>(m_knots1,m_knots2);
	}


	/**
	 * @class b-spline surface calculation class, first and second directions partial specialization
	 * @brief Class for the calculation of a b-spline surface value at a given point in parametric space given a set of knots and control points
	 * @tparam P1 b-spline basis order (first direction)
	 * @tparam N1 number of basis functions of order P1
	 * @tparam N1 number of basis functions of order P2
	 * @tparam DIM co-domain number of dimensions
	 */
	// Use variadic templates
	template<int N1, int N2, int DIM>
	class BSplineSurface<0,N1,0,N2,DIM> : public PolinomialParametricSurface<DIM>
	{
	public:

		// TODO: Use single container for points, knots, control points, etc..
		// TODO: add static assert to check values of template parameters
		// TODO: switch to GridTools data types (u_int, etc)

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set (first direction)
		 * @param i_knots2 b-spline basis knot set (first direction)
		 * @param i_controlPoints control point set in R^DIM
		 */
		BSplineSurface(const std::array<double,N1+1>& i_knots1, const std::array<double,N2+1> i_knots2, const std::vector<Point<DIM> >& i_controlPoints)
					  :PolinomialParametricSurface<DIM>(i_controlPoints)
					  ,m_knots1(i_knots1)
					  ,m_knots2(i_knots2)
					  {}

		/**
		 * @brief Surface jacobian evaluation method
		 * @param i_csi Parametric space point value
		 * @param i_eta Parametric space point value (second direction)
		 * @return Point in R^DIM (first and second parameter derivatives)
		 */
		inline std::array<Point<DIM>,2> evaluateJacobian(double i_csi, double i_eta) const { return std::array<Point<DIM>,2>{Point<DIM>{},Point<DIM>{}}; }

	private:

		/**
		 * Non copiable class
		 */
		BSplineSurface(const BSplineSurface<0,N1,0,N2,DIM>&);

		/**
		 * Non assignable class
		 */
		BSplineSurface<0,N1,0,N2,DIM>& operator=(const BSplineSurface<0,N1,0,N2,DIM>&);

		/**
		 * @brief basis generation method (to be implemented by derived specific classes)
		 * @return (new) pointer to basis
		 */
		BivariatePolinomialParametricBasis* generateBasis(void) const;

		/**
		 * b-splines basis knots set first direction
		 */
		const std::array<double,N1+1> m_knots1;

		/**
		 * b-splines basis knots set second direction
		 */
		const std::array<double,N2+1> m_knots2;

	};

	template<int N1, int N2, int DIM>
	BivariatePolinomialParametricBasis* BSplineSurface<0,N1,0,N2,DIM>::generateBasis(void) const
	{
		return new BivariateBSplineBasis<0,N1,0,N2>(m_knots1,m_knots2);
	}

}
