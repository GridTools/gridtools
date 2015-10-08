#pragma once

#include <array>
#include "nd_generic_basis_rt.h"

namespace iga_rt
{

	/**
	 * @struct b-spline basis generation structure
	 * @brief Struct for recursive (compile time) generation of b-spline basis given a set of knots
	 * @tparam P b-spline basis order
	 * @tparam N number of basis functions of order P
	 */
	template<int P, int N, int RUN_N=N>
	struct BSplineBasisGenerator
	{
		/**
		 * @brief b-spline generation method
		 * @param i_knots b-spline basis knot set
		 * @param io_bsplines array of pointers to b-spline functions (ordered from 1 to N)
		 */
		static void generateBSplineBasis(const std::array<double,N+P+1>& i_knots, std::array<BaseBSpline*,N>& io_bsplines)// TODO: remove duplicated code
		{
			io_bsplines[RUN_N-1] = new BSpline<RUN_N,P>(i_knots.data());
			BSplineBasisGenerator<P,N,RUN_N-1>::generateBSplineBasis(i_knots, io_bsplines);
		}
	};

	template<int P, int N>
	struct BSplineBasisGenerator<P,N,0>
	{
		static void generateBSplineBasis(const std::array<double,N+P+1>& i_knots, std::array<BaseBSpline*,N>& io_bsplines)
		{
		}
	};

	template<int... Args>
	struct ComputeMultiplicity
	{
		constexpr static int m_multiplicity{1};
	};

	template<int P1, int N1, int... Args>
	struct ComputeMultiplicity<P1,N1,Args...>
	{
		constexpr static int m_multiplicity{N1*ComputeMultiplicity<Args...>::m_multiplicity};
	};

	template<int... Args>
	struct ComputeDimension
	{
		constexpr static int m_dimension{0};
	};

	template<int P1, int N1, int... Args>
	struct ComputeDimension<P1,N1,Args...>
	{
		constexpr static int m_dimension{1 + ComputeDimension<Args...>::m_dimension};
	};

	template<int... Args>
	struct ComputeKnotSetSize
	{
		constexpr static int m_size{0};
	};

	template<int P1, int N1, int... Args>
	struct ComputeKnotSetSize<P1,N1,Args...>
	{
		constexpr static int m_size{(N1 + P1 + 1) + ComputeKnotSetSize<Args...>::m_size};
	};

	template<int P1, int N1, int... Args>
	struct ComputeDerivativesDataSize
	{
		constexpr static int m_size{ComputeMultiplicity<P1,N1,Args...>::m_multiplicity*ComputeDimension<P1,N1,Args...>::m_dimension};
	};


	template<typename T, int L, int I_START, int I_STOP>
	struct SubArrayCreator
	{
		static std::array<T,I_STOP - I_START> create(const std::array<T,L>& i_array)
		{
			std::array<T,I_STOP - I_START> o_subArray;

			for(unsigned int i=0;i<I_STOP - I_START;++i)
			{
				o_subArray[i] = i_array[I_START + i];

			}
			return o_subArray;
		}
	};


	template<typename T, int I_START, int I_STOP>
	struct SubArrayCreator<T,0,I_START,I_STOP>
	{
		static std::array<T,0> create(const std::array<T,0>& i_array)
		{
			std::array<T,0> o_emptyArray;
			return o_emptyArray;
		}
	};


	// TODO: update doxy
	template<int... Args>
	class NDBSplineBasis;

	template<int P1, int N1, int... Args>
	class NDBSplineBasis<P1,N1,Args...> : public NDPolinomialParametricBasis<ComputeDimension<P1,N1,Args...>::m_dimension,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity>
	{
	public:

		// TODO: add static assert to check values of template parameters
		// TODO: implement b_spline basis function evaluation selection with
		//		 node span method
		// TODO: introduce alias for sizes and dimensions in public interface (cleaner code and maybe save compile time?)

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 */
		NDBSplineBasis(const std::array<double,ComputeKnotSetSize<P1,N1,Args...>::m_size>& i_knots);

		/**
		 * @brief Destructor
		 */
		virtual ~NDBSplineBasis(void);

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
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
		 * @brief Knot span finding method, returning the knot index i such as csi_{i}<=i_csi<csi_{i+1}
		 * @param i_csi Parametric space point value
		 * @return knot index respecting the condition described above
		 */
		std::array<unsigned int,ComputeDimension<P1,N1,Args...>::m_dimension> find_span(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const;

	private:

		/**
		 * Non copyable class
		 */
		NDBSplineBasis(const NDBSplineBasis<P1,N1,Args...>&);

		/**
		 * Non assignable class
		 */
		NDBSplineBasis<P1,N1,Args...>& operator=(const NDBSplineBasis<P1,N1,Args...>&);

		// TOD : doxy
		std::array<double,N1> evaluateLocalBasis(double i_csi) const;

		/**
		 * b-spline base function set
		 */
		std::array<BaseBSpline*,N1> m_bsplines; // TODO: this should be const
											  // TODO: a generic function set object should be put in generic basis, where evaluation loops should be implemented

		/**
		 * b-spline basis knot set number
		 */
		constexpr static unsigned int m_number_of_knots{ComputeKnotSetSize<P1,N1>::m_size};

		/**
		 * b-spline basis knot set
		 */
		const std::array<double,m_number_of_knots> m_knots;

		/**
		 * b-spline basis knot set last knot multiplicity status (required for correct calculation of basis value on knot set closure, multiplicity should be P+1 for interpolant basis)
		 */
		const bool m_isLastKnotRepeated;

		NDBSplineBasis<Args...> m_residualBasis;

	};

	template<int P1, int N1, int... Args>
	NDBSplineBasis<P1,N1,Args...>::NDBSplineBasis(const std::array<double,ComputeKnotSetSize<P1,N1,Args...>::m_size>& i_knots)
											     :m_knots(SubArrayCreator<double,ComputeKnotSetSize<P1,N1,Args...>::m_size,0,ComputeKnotSetSize<P1,N1>::m_size>::create(i_knots))
											     ,m_isLastKnotRepeated((i_knots[N1+P1]==i_knots[N1]) ? true : false)// TODO: this check is based on the condition i_knots[i]<=i_knots[i+1] which is not checked anywhere
											     ,m_residualBasis(SubArrayCreator<double,ComputeKnotSetSize<P1,N1,Args...>::m_size,ComputeKnotSetSize<P1,N1>::m_size,ComputeKnotSetSize<P1,N1>::m_size + ComputeKnotSetSize<Args...>::m_size>::create(i_knots))
	{
		// TODO: add check on number of nodes
		// TODO: update interfaces also for other dimensions
		// TODO: do we have some ad-hoc container in grid tools for array+size?
		BSplineBasisGenerator<P1,N1>::generateBSplineBasis(m_knots, m_bsplines);
	}


	template<int P1, int N1, int... Args>
	NDBSplineBasis<P1,N1,Args...>::~NDBSplineBasis(void)
	{
		// TODO: stl algos
		for(unsigned int i=0;i<N1;++i)
		{
			delete m_bsplines[i];
		}
	}

	template<int P1, int N1, int... Args>
	std::array<double,N1> NDBSplineBasis<P1,N1,Args...>::evaluateLocalBasis(const double i_csi) const
	{
		// Evaluate local basis
		std::array<double,N1> o_local_values;
		// TODO: use iterator
		for(int i=0;i<N1-1;++i)
		{
			o_local_values[i] = m_bsplines[i]->evaluate(i_csi);
		}

		if(m_isLastKnotRepeated && i_csi == m_knots[N1+P1])
		{
			o_local_values[N1-1] = 1.;
		}
		else
		{
			o_local_values[N1-1] = m_bsplines[N1-1]->evaluate(i_csi);
		}

		return o_local_values;
	}


	template<int P1, int N1, int... Args>
	std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> NDBSplineBasis<P1,N1,Args...>::evaluate(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const
	{
		// TODO: this is not efficient since it requires the creation/copy of N arrays

		// Evaluate local basis
		const std::array<double,N1> local_values(evaluateLocalBasis(i_csi[0]));

		// Evaluate residual basis
		const std::array<double,ComputeMultiplicity<Args...>::m_multiplicity> residual_values(m_residualBasis.evaluate(SubArrayCreator<double,ComputeDimension<P1,N1,Args...>::m_dimension,1,1 + ComputeDimension<Args...>::m_dimension>::create(i_csi)));

		// TODO: use iterator
		std::array<double,ComputeMultiplicity<P1,N1,Args...>::m_multiplicity> o_values;
		int global_index(0);
		for(int i=0;i<N1;++i)
		{
			for(int j=0;j<residual_values.size();++j,++global_index)
			{
				o_values[global_index] = local_values[i]*residual_values[j];
			}
		}

		return o_values;
	}

	/**
	 * @brief base functions derivative evaluation method for a given point in parametric space
	 * @param i_csi Parametric space point value
	 * @return Values corresponding to base function derivative values at the provided point
	 */
	template<int P1, int N1, int... Args>
	std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> NDBSplineBasis<P1,N1,Args...>::evaluateDerivatives(const std::array<double,ComputeDimension<P1,N1,Args...>::m_dimension>& i_csi) const
	{
		// TODO: this is not efficient since it requires the creation/copy of N arrays

		// Compute local basis function values
		const std::array<double,N1> local_values(evaluateLocalBasis(i_csi[0]));

		// Compute local basis function derivative values
		std::array<double,ComputeMultiplicity<P1,N1>::m_multiplicity> local_derivatives_values;
		// TODO: use iterator
		// TODO: what about derivative on last knot? (as for function value)
		// TODO: factorize with evaluate method
		for(int i=0;i<N1;++i)
		{
			local_derivatives_values[i] = m_bsplines[i]->evaluateDerivatives(i_csi[0]);
		}

		// Compute residual basis function values
		const std::array<double,ComputeMultiplicity<Args...>::m_multiplicity> residual_values(m_residualBasis.evaluate(SubArrayCreator<double,ComputeDimension<P1,N1,Args...>::m_dimension,1,1 + ComputeDimension<Args...>::m_dimension>::create(i_csi)));

		// Compute residual basis function derivative values
		const std::array<double,ComputeDerivativesDataSize<Args...>::m_size> residual_derivatives_values(m_residualBasis.evaluateDerivatives(SubArrayCreator<double,ComputeDimension<P1,N1,Args...>::m_dimension,1,1 + ComputeDimension<Args...>::m_dimension>::create(i_csi)));

		// Build global derivative set
		// TODO: use iterator
		std::array<double,ComputeDerivativesDataSize<P1,N1,Args...>::m_size> o_derivatives_values;
		int global_index(0);
		for(int i=0;i<N1;++i)
		{
			for(int j=0;j<residual_values.size();++j,++global_index)
			{
				o_derivatives_values[global_index] = local_derivatives_values[i]*residual_values[j];
			}
		}

		for(int i=0;i<N1;++i)
		{
			for(int j=0;j<residual_derivatives_values.size();++j,++global_index)
			{
				o_derivatives_values[global_index] = local_values[i]*residual_derivatives_values[j];
			}
		}

		return o_derivatives_values;
	}


	template<int P1, int N1>
	class NDBSplineBasis<P1,N1> : public NDPolinomialParametricBasis<ComputeDimension<P1,N1>::m_dimension,ComputeMultiplicity<P1,N1>::m_multiplicity>
	{
	public:

		// TODO: add static assert to check values of template parameters
		// TODO: implement b_spline basis function evaluation selection with
		//		 node span method
		// TODO: introduce alias for sizes and dimensions in public interface (cleaner code and maybe save compile time?)

		/**
		 * @brief Constructor
		 * @param i_knots b-spline basis knot set
		 */
		NDBSplineBasis(const std::array<double,ComputeKnotSetSize<P1,N1>::m_size>& i_knots);

		/**
		 * @brief Destructor
		 */
		virtual ~NDBSplineBasis(void);

		/**
		 * @brief b-spline base functions evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return (N) values corresponding to base function values at the provided point
		 */
		std::array<double,ComputeMultiplicity<P1,N1>::m_multiplicity> evaluate(const std::array<double,ComputeDimension<P1,N1>::m_dimension>& i_csi) const;

		/**
		 * @brief base functions derivative evaluation method for a given point in parametric space
		 * @param i_csi Parametric space point value
		 * @return Values corresponding to base function derivative values at the provided point
		 */
		std::array<double,ComputeDerivativesDataSize<P1,N1>::m_size> evaluateDerivatives(const std::array<double,ComputeDimension<P1,N1>::m_dimension>& i_csi) const;

		/**
		 * @brief Knot span finding method, returning the knot index i such as csi_{i}<=i_csi<csi_{i+1}
		 * @param i_csi Parametric space point value
		 * @return knot index respecting the condition described above
		 */
		std::array<unsigned int,ComputeDimension<P1,N1>::m_dimension> find_span(const std::array<double,ComputeDimension<P1,N1>::m_dimension>& i_csi) const;

	private:

		/**
		 * Non copyable class
		 */
		NDBSplineBasis(const NDBSplineBasis<P1,N1>&);

		/**
		 * Non assignable class
		 */
		NDBSplineBasis<P1,N1>& operator=(const NDBSplineBasis<P1,N1>&);


		/**
		 * b-spline base function set
		 */
		std::array<BaseBSpline*,N1> m_bsplines; // TODO: this should be const
											  // TODO: a generic function set object should be put in generic basis, where evaluation loops should be implemented

		/**
		 * b-spline basis knot set number
		 */
		constexpr static unsigned int m_number_of_knots{ComputeKnotSetSize<P1,N1>::m_size};

		/**
		 * b-spline basis knot set
		 */
		const std::array<double,m_number_of_knots> m_knots;

		/**
		 * b-spline basis knot set last knot multiplicity status (required for correct calculation of basis value on knot set closure, multiplicity should be P+1 for interpolant basis)
		 */
		const bool m_isLastKnotRepeated;

	};

	template<int P1, int N1>
	NDBSplineBasis<P1,N1>::NDBSplineBasis(const std::array<double,ComputeKnotSetSize<P1,N1>::m_size>& i_knots)
										 :m_knots(SubArrayCreator<double,ComputeKnotSetSize<P1,N1>::m_size,0,ComputeKnotSetSize<P1,N1>::m_size>::create(i_knots))
										 ,m_isLastKnotRepeated((i_knots[N1+P1]==i_knots[N1]) ? true : false)// TODO: this check is based on the condition i_knots[i]<=i_knots[i+1] which is not checked anywhere
	{
		// TODO: add check on number of nodes
		// TODO: update interfaces also for other dimensions
		// TODO: do we have some ad-hoc container in grid tools for array+size?
		BSplineBasisGenerator<P1,N1>::generateBSplineBasis(m_knots, m_bsplines);
	}


	template<int P1, int N1>
	NDBSplineBasis<P1,N1>::~NDBSplineBasis(void)
	{
		// TODO: stl algos
		for(unsigned int i=0;i<N1;++i)
		{
			delete m_bsplines[i];
		}
	}

	template<int P1, int N1>
	std::array<double,ComputeMultiplicity<P1,N1>::m_multiplicity> NDBSplineBasis<P1,N1>::evaluate(const std::array<double,ComputeDimension<P1,N1>::m_dimension>& i_csi) const
	{
		// TODO: this is inefficients since it requires the creation/copy of N arrays

		// Evaluate local basis
		std::array<double,N1> o_values;
		// TODO: use iterator
		for(int i=0;i<N1-1;++i)
		{
			o_values[i] = m_bsplines[i]->evaluate(i_csi[0]);
		}

		if(m_isLastKnotRepeated && i_csi[0] == m_knots[N1+P1])
		{
			o_values[N1-1] = 1.;
		}
		else
		{
			o_values[N1-1] = m_bsplines[N1-1]->evaluate(i_csi[0]);
		}

		return o_values;
	}

	template<int P1, int N1>
	std::array<double,ComputeDerivativesDataSize<P1,N1>::m_size> NDBSplineBasis<P1,N1>::evaluateDerivatives(const std::array<double,ComputeDimension<P1,N1>::m_dimension>& i_csi) const
	{
		std::array<double,N1> o_values;

		// TODO: use iterator
		// TODO: what about derivative on last knot? (as for function value)
		// TODO: factorize with evaluate method
		for(int i=0;i<N1;++i)
		{
			o_values[i] = m_bsplines[i]->evaluateDerivatives(i_csi[0]);
		}

		return o_values;
	}
}
