# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2019.                                    *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_VARIADIC_HAS_OPT_HPP
# define GT_PREPROCESSOR_VARIADIC_HAS_OPT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_VARIADIC_HAS_OPT */
#
# if defined(__cplusplus) && __cplusplus > 201703L
#  if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 10
#   define GT_PP_VARIADIC_HAS_OPT() 0
#  elif defined(__clang__) && __clang_major__ < 9
#   define GT_PP_VARIADIC_HAS_OPT() 0
#  else
#   include <gridtools/preprocessor/variadic/detail/has_opt.hpp>
#   define GT_PP_VARIADIC_HAS_OPT() \
  GT_PP_VARIADIC_HAS_OPT_ELEM2(GT_PP_VARIADIC_HAS_OPT_FUNCTION(?),) \
/**/
#  endif
# else
# define GT_PP_VARIADIC_HAS_OPT() 0
# endif
#
# endif
