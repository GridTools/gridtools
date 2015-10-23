#pragma once
#include <array>
#include <numeric>
#include "nd_generic_basis_rt.h"
#include "nd_b_splines_basis_rt.h"

namespace iga_rt
{

	template<int... Args>
	class NDNurbsBasis;

	// TODO: update doxy
	/**
	 * @class Nurbs basis class
	 * @brief Class for the composition of a nurbs basis given a set of knots and a set of weights
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions
	 */
	template<int P1, int N1, int... Args>
	class NDNurbsBasis<P1,N1,Args...> : public NDPolinomialParametricBasis<ComputeDimension<P1,N1,Args...>::m_dimension,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity>
	{
	public:

		// TODO: check on input parameters value required (e.g., non zero weight) as well as on input array size

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 * @param i_weights b-spline basis function linear combination weight set
		 */
		NDNurbsBasis(const std::array<double,ComputeKnotSetSize<P1,N1,Args...>::m_size>& i_knots,
					 const std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity>& i_weights)
					:m_bspline_basis(i_knots)
					,m_weights(i_weights)
					,m_normalization_factor(0.)
		{}

		/**
		 * @brief nurbs base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> evaluate(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const;

		/**
		 * @brief base functions derivative evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function derivative values at the provided point
		 */
		std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> evaluateDerivatives(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const;

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
		NDNurbsBasis(const NDNurbsBasis<P1,N1,Args...>&);

		/**
		 * Non assignable class
		 */
		NDNurbsBasis<P1,N1,Args...>& operator=(const NDNurbsBasis<P1,N1,Args...>&);

		/**
		 * b-spline basis
		 */
		const NDBSplineBasis<P1,N1,Args...> m_bspline_basis;

		/**
		 * Weight list
		 */
		const std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> m_weights;

		/**
		 * Normalization factor corresponding to last basis evaluation
		 */
		mutable double m_normalization_factor;// TODO: avoid mutable

	};

	template<int P1, int N1, int... Args>
	std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> NDNurbsBasis<P1,N1,Args...>::evaluate(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const
	{
		// Compute b-spline basis function values for the provided parameter space point
		std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> b_spline_values(m_bspline_basis.evaluate(i_csi));

		// Compute b-splines weightening
		// TODO: switch to std algos
		for(unsigned int basis_index=0;basis_index<ComputeMultiplicity<P1,N1,Args...>::m_multiplicity;++basis_index)
		{
			b_spline_values[basis_index] *= m_weights[basis_index];
		}

		// Compute weighted sum reciprocal (nurbs denominator)
		m_normalization_factor = 1./std::accumulate(b_spline_values.begin(),b_spline_values.end(),0.);

		// Compute Nurbs basis functions
		// TODO: switch to std algos
		std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> o_nurbs_basis;
		for(unsigned int basis_index=0;basis_index<ComputeMultiplicity<P1,N1,Args...>::m_multiplicity;++basis_index)
		{
			o_nurbs_basis[basis_index] = b_spline_values[basis_index]*m_normalization_factor;
		}

		return o_nurbs_basis;
	}

	// TODO: factorize with evaluate method
	// TODO: are evaluate and evaluateDerivative methods movable to base class?
	template<int P1, int N1, int... Args>
	std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> NDNurbsBasis<P1,N1,Args...>::evaluateDerivatives(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const
	{
		// Compute nurbs values // TODO: this is already computed by evaluate method!
		const std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> nurbs_values(evaluate(i_csi));

		// Compute b-spline derivatives values
		std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> b_splines_derivatives_values(m_bspline_basis.evaluateDerivatives(i_csi));

		// Compute derivative weightening
		// TODO: switch to std algos
		for(unsigned int var_index=0;var_index<ComputeDimension<P1,N1,Args...>::m_dimension;++var_index)
		{
			for(unsigned int basis_index=0;basis_index<ComputeMultiplicity<P1,N1,Args...>::m_multiplicity;++basis_index)
			{
				b_splines_derivatives_values[basis_index + var_index*ComputeMultiplicity<P1,N1,Args...>::m_multiplicity] *= m_weights[basis_index];
			}
		}

		// Compute normalization factor derivative
		const double normalization_factor_der = std::accumulate(b_splines_derivatives_values.begin(),b_splines_derivatives_values.end(),0.);

		// Compute derivative values
		// TODO: switch to stl algos
		std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> o_derivative_values;
		for(unsigned int var_index=0;var_index<ComputeDimension<P1,N1,Args...>::m_dimension;++var_index)
		{
			for(unsigned int basis_index=0;basis_index<ComputeMultiplicity<P1,N1,Args...>::m_multiplicity;++basis_index)
			{
				o_derivative_values[basis_index + var_index*ComputeMultiplicity<P1,N1,Args...>::m_multiplicity] = (b_splines_derivatives_values[basis_index + var_index*ComputeMultiplicity<P1,N1,Args...>::m_multiplicity]-nurbs_values[basis_index]*normalization_factor_der)*m_normalization_factor;
			}
		}

		return o_derivative_values;

	}

	template<int P1, int N1, int... Args>
	double NDNurbsBasis<P1,N1,Args...>::getNormalizationFactor(void) const
	{
		if(m_normalization_factor==0.)
		{
			//TODO: add exception
		}

		return m_normalization_factor;
	}
}
