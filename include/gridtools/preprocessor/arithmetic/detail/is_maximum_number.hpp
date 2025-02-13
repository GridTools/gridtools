# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2020.                                    *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_ARITHMETIC_DETAIL_IS_MAXIMUM_NUMBER_HPP
# define GT_PREPROCESSOR_ARITHMETIC_DETAIL_IS_MAXIMUM_NUMBER_HPP
#
# /* GT_PP_DETAIL_IS_MAXIMUM_NUMBER */
#
# include <gridtools/preprocessor/comparison/equal.hpp>
# include <gridtools/preprocessor/arithmetic/detail/maximum_number.hpp>
#
# define GT_PP_DETAIL_IS_MAXIMUM_NUMBER(x) GT_PP_EQUAL(x,GT_PP_DETAIL_MAXIMUM_NUMBER)
#
# endif
