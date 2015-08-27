#pragma once
#include <array>

namespace iga_rt
{

	/**
	 * @class Point in R^DIM space representation struct
	 * @brief Struct for the representation of point in R^DIM
	 * @tparam DIM number of space dimensions
	 */
	template <int DIM>
	class Point
	{
	public:

		Point(void)
		{
			m_coords.fill(0.);
		}

		/**
		 * @brief Operator += overloading for sum of points in R^DIM
		 * @param i_point Point to be summed
		 * @return Sum result
		 */
		Point<DIM>& operator+=(const Point<DIM>& i_point);

		/**
		 * @brief Operator *= overloading for scalar and point in R^DIM multiplication
		 * @param i_factor Scalar factor to be multiplied
		 * @return Multiplication result
		 */
		Point<DIM>& operator*=(double i_factor);

		/**
		 * Point coordinate set
		 */
		std::array<double,DIM> m_coords;
	};

	template <int DIM>
	Point<DIM>& Point<DIM>::operator+=(const Point<DIM>& i_point)
	{
		// TODO: use std algos
		for(int i=0;i<DIM;++i)
		{
			m_coords[i] += i_point.m_coords[i];
		}
		return *this;
	}

	template <int DIM>
	Point<DIM>& Point<DIM>::operator*=(const double i_factor)
	{
		// TODO: use std algos
		for(int i=0;i<DIM;++i)
		{
			m_coords[i] *= i_factor;
		}
		return *this;
	}

	/**
	 * @brief Operator + overloading for sum of points in R^DIM
	 * @tparam DIM number of space dimensions
	 * @param i_point1 First addend
	 * @param i_point2 Second addend
	 * @return Point sum result
	 */
	template <int DIM>
	Point<DIM> operator+(const Point<DIM>& i_point1, const Point<DIM>& i_point2)
	{
		Point<DIM> o_point;
		// TODO: use contract operators
		// TODO: use std algos
		// TODO: avoid temporary copy
		for(int i=0;i<DIM;++i)
		{
			o_point.m_coords[i] = i_point1.m_coords[i] + i_point2.m_coords[i];
		}
		return o_point;
	}

	/**
	 * @brief Operator - overloading for sum of points in R^DIM
	 * @tparam DIM number of space dimensions
	 * @param i_point1 First addend
	 * @param i_point2 Second addend
	 * @return Point difference result
	 */
	template <int DIM>
	Point<DIM> operator-(const Point<DIM>& i_point1, const Point<DIM>& i_point2)
	{
		Point<DIM> o_point;
		// TODO: use contract operators
		// TODO: use std algos
		// TODO: avoid temporary copy
		for(int i=0;i<DIM;++i)
		{
			o_point.m_coords[i] = i_point1.m_coords[i] - i_point2.m_coords[i];
		}
		return o_point;
	}

	/**
	 * @brief Operator * overloading for scalar and point in R^DIM multiplication
	 * @tparam DIM number of space dimensions
	 * @param i_factor Multiplication scalar factor
	 * @param i_point Multiplication point in R^DIM
	 * @return Multiplication result
	 */
	template <int DIM>
	Point<DIM> operator*(const double i_factor, const Point<DIM>& i_point)
	{
		Point<DIM> o_point;
		// TODO: use contract operators
		// TODO: use std algos
		for(int i=0;i<DIM;++i)
		{
			o_point.m_coords[i] = i_factor*i_point.m_coords[i];
		}
		return o_point;
	}

	/**
	 * @brief Operator / overloading for scalar and point in R^DIM multiplication
	 * @tparam DIM number of space dimensions
	 * @param i_point Ratio point in R^DIM
	 * @param i_factor Ratio scalar factor
	 * @return Ratio result
	 */
	template <int DIM>
	Point<DIM> operator/(const Point<DIM>& i_point,const double i_factor)
	{
		Point<DIM> o_point;
		// TODO: use contract operators
		// TODO: use std algos
		for(int i=0;i<DIM;++i)
		{
			o_point.m_coords[i] = i_point.m_coords[i]/i_factor;
		}
		return o_point;
	}

}
