# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2019.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_FACILITIES_CHECK_EMPTY_HPP
# define GT_PREPROCESSOR_FACILITIES_CHECK_EMPTY_HPP
# include <gridtools/preprocessor/variadic/has_opt.hpp>
# if GT_PP_VARIADIC_HAS_OPT()
# include <gridtools/preprocessor/facilities/is_empty_variadic.hpp>
# define GT_PP_CHECK_EMPTY(...) GT_PP_IS_EMPTY_OPT(__VA_ARGS__)
# endif /* GT_PP_VARIADIC_HAS_OPT() */
# endif /* GT_PREPROCESSOR_FACILITIES_CHECK_EMPTY_HPP */
