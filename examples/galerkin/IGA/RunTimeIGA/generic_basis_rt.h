#pragma once

namespace iga_rt
{
	// TODO: change class names, they are just the interfaces for generic basis functions (or functions in R^N)
	// TODO: implement a new abstraction level for basis making use of knots
// TODO: implement a new abstraction level for basis making use of knots and weights

	/**
	 * @class Generic function basis composition base class
	 * @brief Base class for the composition generic function basis
	 */
	class PolinomialParametricBasis
	{
	public:

		// TODO: use variadic template to generalize to different dimensions (or update other classes with constexpr, etc..)

		/**
		 * @brief Constructor
		 */
		PolinomialParametricBasis(void){};

		virtual ~PolinomialParametricBasis(void){}

		/**
		 * @brief base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function values at the provided point
		 */
		virtual std::vector<double> evaluate(double i_csi) const = 0;

	};


	/**
	 * @class Generic function basis composition base class
	 * @brief Base class for the composition of generic function basis
	 */
	class BivariatePolinomialParametricBasis
	{
	public:

		// TODO: use variadic template to generalize to different dimensions (or update other classes with constexpr, etc..)

		/**
		 * @brief Constructor
		 */
		BivariatePolinomialParametricBasis(void){};

		virtual ~BivariatePolinomialParametricBasis(void){}

		/**
		 * @brief base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value (first direction)
		 * @param i_eta Parametric space point value (second direction)
		 * @return Values corresponding to base function values at the provided point
		 */
		virtual std::vector<double> evaluate(double i_csi, double i_eta) const = 0;

	};

}
