#pragma once
#include <vector>
#include <array>
#include <numeric>
#include "generic_basis_rt.h"
#include "b_splines_basis_rt.h"

namespace iga_rt
{

	/**
	 * @class Nurbs basis class
	 * @brief Class for the composition of a nurbs basis given a set of knots and a set of weights
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions
	 */
	template<int P, int N>
	class NurbsBasis : public PolinomialParametricBasis
	{
	public:

		// TODO: check on input parameters value required (e.g., non zero weight) as well as on input array size

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 * @param i_weights b-spline basis function linear combination weight set
		 */
		NurbsBasis(const std::array<double,N+P+1>& i_knots, const double* i_weights)
			 :m_bspline_basis(i_knots)
			 ,m_weights(i_weights)
			 ,m_normalization_factor(0.)
		{}

		/**
		 * @brief nurbs base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi) const;

		/**
		 * @brief base functions derivative evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function derivative values at the provided point
		 */
		std::vector<double> evaluateDerivatives(double i_csi) const;

		/**
		 * @brief nurbs base functions normalization factor retrieval method (corresponding to last basis evaluation)
		 * @return normalization factor corresponding to last basis evaluation
		 * @throw //TODO: what? if no previous basis evaluation has been requested
		 */
		double getNormalizationFactor(void) const;


	private:

		/**
		 * Non copyable class
		 */
		NurbsBasis(const NurbsBasis<P,N>&);

		/**
		 * Non assignable class
		 */
		NurbsBasis<P,N>& operator=(const NurbsBasis<P,N>&);

		/**
		 * b-spline basis
		 */
		const BSplineBasis<P,N> m_bspline_basis;

		/**
		 * Weight list
		 */
		const double* m_weights;

		/**
		 * Normalization factor corresponding to last basis evaluation
		 */
		mutable double m_normalization_factor;// TODO: avoid mutable

	};

	template<int P, int N>
	std::vector<double> NurbsBasis<P,N>::evaluate(const double i_csi) const
	{
		// Compute b-spline basis function values for the provided parameter space point
		std::vector<double> b_spline_values(m_bspline_basis.evaluate(i_csi));

		// Compute b-splines weightening
		// TODO: switch to std algos
		for(unsigned int basis_index=0;basis_index<N;++basis_index)
		{
			b_spline_values[basis_index] *= m_weights[basis_index];
		}

		// Compute weighted sum reciprocal (nurbs denominator)
		m_normalization_factor = 1./std::accumulate(b_spline_values.begin(),b_spline_values.end(),0.);

		// Compute Nurbs basis functions
		// TODO: switch to std algos
		std::vector<double> o_nurbs_basis(N);
		for(unsigned int basis_index=0;basis_index<N;++basis_index)
		{
			o_nurbs_basis[basis_index] = b_spline_values[basis_index]*m_normalization_factor;
		}

		return o_nurbs_basis;
	}

	template<int P, int N>
	std::vector<double> NurbsBasis<P,N>::evaluateDerivatives(const double i_csi) const
	{
		// Compute nurbs values // TODO: this is already computed by evaluate method!
		const std::vector<double> nurbs_values(evaluate(i_csi));

		// Compute b-spline derivatives values
		std::vector<double> b_splines_derivatives_values(m_bspline_basis.evaluateDerivatives(i_csi));

		// Compute derivative weightening
		// TODO: switch to std algos
		for(unsigned int basis_index=0;basis_index<N;++basis_index)
		{
			b_splines_derivatives_values[basis_index] *= m_weights[basis_index];
		}

		// Compute normalization factor derivative
		const double normalization_factor_der = std::accumulate(b_splines_derivatives_values.begin(),b_splines_derivatives_values.end(),0.);

		// Compute derivative values
		// TODO: switch to stl algos
		std::vector<double> o_derivative_values(N,0.);
		for(unsigned int basis_index=0;basis_index<N;++basis_index)
		{
			o_derivative_values[basis_index] = (b_splines_derivatives_values[basis_index]-nurbs_values[basis_index]*normalization_factor_der)*m_normalization_factor;
		}

		return o_derivative_values;
	}


	template<int P, int N>
	double NurbsBasis<P,N>::getNormalizationFactor(void) const
	{
		if(m_normalization_factor==0.)
		{
			//TODO: add exception
		}

		return m_normalization_factor;
	}

	// TODO: as for b-spline the 1D, 2D, ..., ND cases can be unified
	/**
	 * @class Bivariate Nurbs basis class
	 * @brief Class for the composition of a bivariate nurbs basis given a set of knots and a set of weights
	 * @tparam P1 b-spline basis order in first direction
	 * @tparam N1 number of basis functions of order P1
	 * @tparam P2 b-spline basis order in second direction
	 * @tparam N2 number of basis functions of order P2
	 */
	template<int P1, int N1, int P2, int N2>
	class BivariateNurbsBasis : public BivariatePolinomialParametricBasis
	{
	public:

		/**
		 * @brief Constructor
		 * @param i_knots1 b-spline basis knot set in first direction
		 * @param i_knots2 b-spline basis knot set in second direction
		 * @param i_weights b-spline basis function linear compbination weight set
		 */
		BivariateNurbsBasis(const std::array<double,N1+P1+1>& i_knots1, const std::array<double,N2+P2+1>& i_knots2, const double* i_weights)
						   :m_bspline_basis(i_knots1,i_knots2)
						   ,m_weights(i_weights)
						   ,m_normalization_factor(0.)
		{}

		/**
		 * @brief nurbs base functions evaluation method for a given point pair in parametric space
		 * @param i_csi Parametric space point value in first direction
		 * @param i_eta Parametric space point value in second direction
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::vector<double> evaluate(double i_csi, double i_eta) const;

		/**
		 * @brief base functions derivative evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return Values corresponding to base function derivative values at the provided point (first array component corresponds to derivatives vs first variable)
		 */
		std::array<std::vector<double>,2> evaluateDerivatives(double i_csi, double i_eta) const;

		/**
		 * @brief nurbs base functions normalization factor retrieval method (corresponding to last basis evaluation)
		 * @return normalization factor corresponding to last basis evaluation
		 * @throw //TODO: what? if no previous basis evaluation has been requested
		 */
		double getNormalizationFactor(void) const;

	private:

		/**
		 * Non copyable class
		 */
		BivariateNurbsBasis(const BivariateNurbsBasis<P1,N1,P2,N2>&);

		/**
		 * Non assignable class
		 */
		BivariateNurbsBasis<P1,N1,P2,N2>& operator=(const BivariateNurbsBasis<P1,N1,P2,N2>&);

		/**
		 * b-spline basis
		 */
		const BivariateBSplineBasis<P1,N1,P2,N2> m_bspline_basis;

		/**
		 * Weight list
		 */
		const double* m_weights;

		/**
		 * Normalization factor corresponding to last basis evaluation
		 */
		mutable double m_normalization_factor;// TODO: avoid mutable

	};

	// TODO: code duplication with 1D case
	template<int P1, int N1, int P2, int N2>
	std::vector<double> BivariateNurbsBasis<P1,N1,P2,N2>::evaluate(const double i_csi, const double i_eta) const
	{
		// Compute b-spline basis function values for the provided parameter space point
		std::vector<double> b_spline_values(m_bspline_basis.evaluate(i_csi,i_eta));

		// Compute b-splines weightening
		const unsigned int basis_size(N1*N2);// TODO: switch to constexpr data member
		for(unsigned int basis_index=0;basis_index<basis_size;++basis_index)
		{
			b_spline_values[basis_index] *= m_weights[basis_index];
		}

		// Compute weighted sum reciprocal (nurbs denominator)
		m_normalization_factor = 1./std::accumulate(b_spline_values.begin(),b_spline_values.end(),0.);

		// Compute Nurbs basis functions
		// TODO: switch to std algos
		std::vector<double> o_nurbs_basis(basis_size);
		for(unsigned int basis_index=0;basis_index<basis_size;++basis_index)
		{
			o_nurbs_basis[basis_index] = b_spline_values[basis_index]*m_normalization_factor;
		}

		return o_nurbs_basis;
	}


	// TODO: code duplication with 1D case
	template<int P1, int N1, int P2, int N2>
	std::array<std::vector<double>,2> BivariateNurbsBasis<P1,N1,P2,N2>::evaluateDerivatives(double i_csi, double i_eta) const
	{
		// Compute nurbs values // TODO: this is already computed by evaluate method!
		const std::vector<double> nurbs_values(evaluate(i_csi,i_eta));

		// Compute b-spline derivatives values
		std::array<std::vector<double>,2> b_splines_derivatives_values(m_bspline_basis.evaluateDerivatives(i_csi, i_eta));

		// Compute derivative weightening
		// TODO: switch to std algos
		// TODO: inefficient memory acces pattern
		const unsigned int basis_size(N1*N2);// TODO: switch to constexpr data member
		for(unsigned int basis_index=0;basis_index<basis_size;++basis_index)
		{
			b_splines_derivatives_values[0][basis_index] *= m_weights[basis_index];
			b_splines_derivatives_values[1][basis_index] *= m_weights[basis_index];
		}

		// Compute normalization factor derivatives
		const double normalization_factor_der_first = std::accumulate(b_splines_derivatives_values[0].begin(),b_splines_derivatives_values[0].end(),0.);
		const double normalization_factor_der_second = std::accumulate(b_splines_derivatives_values[1].begin(),b_splines_derivatives_values[1].end(),0.);

		// Compute derivative values
		// TODO: switch to stl algos
		// TODO: inefficient memory acces pattern
		std::array<std::vector<double>,2> o_derivative_values{{std::vector<double>(N1*N2,0.),std::vector<double>(N1*N2,0.)}};
		for(unsigned int basis_index=0;basis_index<basis_size;++basis_index)
		{
			o_derivative_values[0][basis_index] = (b_splines_derivatives_values[0][basis_index]-nurbs_values[basis_index]*normalization_factor_der_first)*m_normalization_factor;
			o_derivative_values[1][basis_index] = (b_splines_derivatives_values[1][basis_index]-nurbs_values[basis_index]*normalization_factor_der_second)*m_normalization_factor;
		}

		return o_derivative_values;
	}


	// TODO: code duplication with 1D case
	template<int P1, int N1, int P2, int N2>
	double BivariateNurbsBasis<P1,N1,P2,N2>::getNormalizationFactor(void) const
	{
		if(m_normalization_factor==0.)
		{
			//TODO: add exception
		}

		return m_normalization_factor;
	}

}
