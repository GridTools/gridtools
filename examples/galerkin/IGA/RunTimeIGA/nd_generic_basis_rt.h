#pragma once

#include <array>

namespace iga_rt
{
	// TODO: change class names, they are just the interfaces for generic basis functions (or functions in R^N)
	// TODO: implement a new abstraction level for basis making use of knots
	// TODO: implement a new abstraction level for basis making use of knots and weights

	/**
	 * @class Generic function basis composition base class
	 * @brief Base class for the composition generic function basis
	 * @tparam D basis function domain number of dimensions
	 * @tparam N basis function number
	 */
	template<int D, int N>
	class NDPolinomialParametricBasis
	{
	public:

		/**
		 * @brief Constructor
		 */
		NDPolinomialParametricBasis(void){};

		virtual ~NDPolinomialParametricBasis(void){}

		/**
		 * @brief base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function values at the provided point
		 */
		virtual std::array<double,N> evaluate(const std::array<double,D>& i_csi) const = 0;

		/**
		 * @brief base functions derivative evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function derivative values at the provided point
		 */
		virtual std::array<double,N*D> evaluateDerivatives(const std::array<double,D>& i_csi) const = 0;

	};
}
