#pragma once

namespace iga_rt
{

	// TODO: implements control points struct in N dim
	// TODO: implements B spline basis composition N dim
	// TODO: implements B spline surface calculation N dim
	// TODO: variadic templates for generic case
	// TODO: switch to structs as in GridTools


	/**
	 * @class b-spline function base class
	 * @brief Pure virtual base class for interface definition only
	 */
	class BaseBSpline
	{
	public:

		virtual ~BaseBSpline(void)
		{}

		/**
		 * @brief b-spline evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value
		 * @return b-spline value
		 */
		virtual double evaluate(double i_csi) const = 0;

		/**
		 * @brief b-spline derivative evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value
		 * @return b-spline derivative value
		 */
		virtual double evaluateDerivatives(double i_csi) const = 0;
	};

	/**
	 * @class Bivariate b-spline function base class
	 * @brief Pure virtual base class for interface definition only
	 */
	class BaseBivariateBSpline
	{
	public:

		// TODO: variadic template to generalize in N-Dimensions

		virtual ~BaseBivariateBSpline(void)
		{}

		/**
		 * @brief b-spline evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return b-spline value
		 */
		virtual double evaluate(double i_csi, double i_eta) const = 0;

		/**
		 * @brief b-spline derivatives evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return b-spline derivative values (first value correspond to the derivative wrt first variable)
		 */
		virtual std::pair<double,double> evaluateDerivatives(double i_csi, double i_eta) const = 0;
	};

	/**
	 * @class Bivariate b-spline function base class
	 * @brief Pure virtual base class for interface definition only
	 */
	class BaseTrivariateBSpline
	{
	public:

		// TODO: variadic template to generalize in N-Dimensions

		virtual ~BaseTrivariateBSpline(void)
		{}

		/**
		 * @brief b-spline evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @param i_gamma Parametric space point value (third direction)
		 * @return b-spline value
		 */
		virtual double evaluate(double i_csi, double i_eta, double i_gamma) const = 0;
	};

	/**
	 * @class b-spline function class
	 * @brief Class for the representation of b-spline function given a set of knots:
	 * 	this class implements the Cox-de Boor recursive formula for b-spline calculation
	 * @tparam I b-spline function index
	 * @tparam P b-spline function order
	 */
	template <int I, int P>
	class BSpline : public BaseBSpline
	{
	public:

		// TODO: add static assert for template parameters check

		/**
		 * @brief Contructor
		 * @param i_knots b-spline function knot set
		 */
            template <typename Array>
            BSpline(const Array* i_knots);

		/**
		 * @param i_csi Parametric space point value
		 * @return b-spline function value
		 */
		double evaluate(double i_csi) const;

		/**
		 * @brief b-spline derivative evaluation method
		 * @param i_csi Parametric space point value
		 * @return b-spline derivative value
		 */
		double evaluateDerivatives(double i_csi) const;

	private:

		/* According to Cox-de Boor formula, the value of a b-spline function
		 * of order P and index I given a set of knots {csi_{i}} with i=1,2,...,N+P+1
		 * can be computed as:
		 *
		 * N_{i,p}(csi) = (csi - csi_{i})*N_{i,p-1}(csi)/(csi_{i+p} - csi_{i}) +
		 * 				  (csi_{i+p+1} - csi)*N_{i+1,p-1}(csi)/(csi_{i+p+1} - csi_{i+1})
		 *
		 * where:
		 * - csi: point in parametric space
		 * - N_{i,p}: b-spline of order p and index i
		 * - csi_{i}: knots at index i
		 */

		/**
		 * csi_{I} knot
		 */
		const double m_csiI;

		/**
		 * csi_{I+1} knot
		 */
		const double m_csiIp1;

		/**
		 * csi_{I+p} knot
		 */
		const double m_csiIpP;

		/**
		 * csi_{I+p+1} knot
		 */
		const double m_csiIpPp1;

		/**
		 * 1/(csi_{I+p}-csi_{I}) Cox-de Boor forumla factor
		 */
		const double m_denIPm1;

		/**
		 * 1/(csi_{I+p+1}-csi_{+1I}) Cox-de Boor forumla factor
		 */
		const double m_denIp1Pm1;

		/**
		 * b-spline function of order P-1 and index I in Cox-de Boor formula
		 */
		BSpline<I,P-1> m_bIPm1;

		/**
		 * b-spline function of order P-1 and index I+1 in Cox-de Boor formula
		 */
		BSpline<I+1,P-1> m_bIp1Pm1;
	};

    template <int I, int P>
    template <typename Array>
    BSpline<I, P>::BSpline(const Array* i_knots)
            :m_csiI((*i_knots)[I-1])
            ,m_csiIp1((*i_knots)[I+1-1])
            ,m_csiIpP((*i_knots)[I+P-1])
            ,m_csiIpPp1((*i_knots)[I+P+1-1])
            ,m_denIPm1((m_csiIpP!=m_csiI)?(1./(m_csiIpP-m_csiI)):0.)
            ,m_denIp1Pm1((m_csiIpPp1!=m_csiIp1)?(1./(m_csiIpPp1-m_csiIp1)):0.)
            ,m_bIPm1(i_knots)
            ,m_bIp1Pm1(i_knots)
	{}


	template <int I, int P>
	double BSpline<I,P>::evaluate(const double i_csi) const
	{
            return (i_csi-m_csiI)*m_denIPm1*m_bIPm1.evaluate(i_csi) + (m_csiIpPp1-i_csi)*m_denIp1Pm1*m_bIp1Pm1.evaluate(i_csi);
	}

	template <int I, int P>
	double BSpline<I,P>::evaluateDerivatives(const double i_csi) const
	{
		// TODO: almost all element of this equation are calculated by evaluate method and can be stored in mutable data member
		return P*m_denIPm1*m_bIPm1.evaluate(i_csi) - P*m_denIp1Pm1*m_bIp1Pm1.evaluate(i_csi);
	}

	/**
	 * @class 0-th order b-spline function class
	 * @brief Class for the representation of 0-th b-spline function given a set of knots
	 * @tparam I b-spline function index
	 */
	template <int I>
	class BSpline<I,0> : public BaseBSpline
	{
	public:

		// TODO: add static assert for template parameters check
                // PAOLO: is accumulate(logical_and(), condition<Pack>...)
                // see accumulate.hpp

		/**
		 * @brief Contructor
		 * @param i_knots b-spline function knot set
		 */
            template <typename Array>
		BSpline(const Array* i_knots)
                    :m_csiI((*i_knots)[I-1])
                    ,m_csiIp1((*i_knots)[I+1-1])
		{}

		/**
		 * @brief b-spline function evaluation method
		 * @param i_csi Parametric space point value
		 * @return b-spline function value
		 */
		double evaluate(const double i_csi) const;

		/**
		 * @brief b-spline derivative evaluation method
		 * @param i_csi Parametric space point value
		 * @return b-spline derivative value
		 */
		double evaluateDerivatives(double i_csi) const;

	private:

		/* According to the definition a 0-th b-spline function of index I given
		 * a set of knots {csi_{i}} with i=1,2,...,N+1 is given by
		 *
		 * N_{i,0}(csi) = 1 if csi_{i}<=csi<csi_{i+1}
		 * and
		 * N_{i,0}(csi) = 0 otherwise
		 *
		 * where:
		 * - csi: point in parametric space
		 * - N_{i,0}: 0-th order b-spline of index i
		 * - csi_{i}: knots at index i
		 */

		/**
		 * csi_{I} knot
		 */
		const double m_csiI;

		/**
		 * csi_{I+1} knot
		 */
		const double m_csiIp1;
	};


	template <int I>
	double BSpline<I,0>::evaluate(const double i_csi) const
	{
            double ret_val(0);
            if(i_csi>=m_csiI && i_csi<m_csiIp1)
		{
			ret_val = 1.;
		}
		else
		{
			ret_val = 0;
		}
            return ret_val;
	}

	template <int I>
	double BSpline<I,0>::evaluateDerivatives(const double i_csi) const
	{
		return 0.;
	}

	/**
	 * @class Bivariate b-spline function class
	 * @brief Class for the representation of bivariate b-spline function given a set of knots:
	 * 	bivariate functions are defined as the tensor product of univariate b-spline functions
	 * @tparam I1 b-spline function index (first direction)
	 * @tparam P1 b-spline function order (first direction)
	 * @tparam I2 b-spline function index (second direction)
	 * @tparam P2 b-spline function order (second direction)
	 */
	template<int I1, int P1, int I2, int P2>
	class BivariateBSpline : public BaseBivariateBSpline
	{
	public:

		// TODO: add static assert for template parameters check
		// TODO: use variadic templates for generic Nvariate bspline

		/**
		 * @brief Contructor
		 * @param i_knots1 b-spline function knot set (first direction)
		 * @param i_knots2 b-spline function knot set (second direction)
		 */
		BivariateBSpline(const double* i_knots1,const double* i_knots2)
						:m_b1(i_knots1)
						,m_b2(i_knots2)
		{}

		/**
		 * @brief b-spline function evaluation method
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return b-spline function value
		 */
		inline double evaluate(const double i_csi, const double i_eta) const { return m_b1.evaluate(i_csi)*m_b2.evaluate(i_eta); }

		/**
		 * @brief b-spline derivatives evaluation method (pure virtual, to be implemented by derived classes)
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return b-spline derivative values (first value correspond to the derivative wrt first variable)
		 */
		std::pair<double,double> evaluateDerivatives(double i_csi, double i_eta) const;

	private:

		/**
		 * b-spline function in first direction
		 */
		const BSpline<I1,P1> m_b1;

		/**
		 * b-spline function in second direction
		 */
		const BSpline<I2,P2> m_b2;
	};


	template<int I1, int P1, int I2, int P2>
	std::pair<double,double> BivariateBSpline<I1,P1,I2,P2>::evaluateDerivatives(double i_csi, double i_eta) const
	{
		// TODO: as for 1D case some element of the following equation are already computed by the evaluate method, use data member to store them
		return std::make_pair(m_b1.evaluateDerivative(i_csi)*m_b2.evaluate(i_eta),
							  m_b1.evaluate(i_csi)*m_b2.evaluateDerivative(i_eta));
	}


	/**
	 * @class Trivariate b-spline function class
	 * @brief Class for the representation of trivariate b-spline function given a set of knots:
	 * 	trivariate functions are defined as the tensor product of univariate b-spline functions
	 * @tparam I1 b-spline function index (first direction)
	 * @tparam P1 b-spline function order (first direction)
	 * @tparam I2 b-spline function index (second direction)
	 * @tparam P2 b-spline function order (second direction)
	 * @tparam I3 b-spline function index (third direction)
	 * @tparam P3 b-spline function order (third direction)
	 */
	template<int I1, int P1, int I2, int P2, int I3, int P3>
	class TrivariateBSpline : public BaseTrivariateBSpline
	{
	public:

		// TODO: use variadic templates for generic Nvariate bspline
		// TODO: add static assert for template parameters check

		/**
		 * @brief Contructor
		 * @param i_knots1 b-spline function knot set (first direction)
		 * @param i_knots2 b-spline function knot set (second direction)
		 * @param i_knots3 b-spline function knot set (third direction)
		 */
		TrivariateBSpline(const double* i_knots1,const double* i_knots2,const double* i_knots3)
						:m_b1(i_knots1,i_knots2)
						,m_b2(i_knots3)
		{}

		/**
		 * @brief b-spline function evaluation method
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @param i_gamma Parametric space point value (third direction)
		 * @return b-spline function value
		 */
		inline double evaluate(double i_csi, double i_eta, double i_gamma) const { return m_b1.evaluate(i_csi,i_eta)*m_b2.evaluate(i_gamma); }

	private:

		/**
		 * Bivariate b-spline function in first-second direction
		 */
		BivariateBSpline<I1,P1,I2,P2> m_b1;

		/**
		 * b-spline function in third direction
		 */
		BSpline<I3,P3> m_b2;
	};
}
