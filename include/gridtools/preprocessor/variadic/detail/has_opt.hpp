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
# ifndef GT_PREPROCESSOR_VARIADIC_DETAIL_HAS_OPT_HPP
# define GT_PREPROCESSOR_VARIADIC_DETAIL_HAS_OPT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if defined(__cplusplus) && __cplusplus > 201703L
#
# if GT_PP_VARIADICS_MSVC
# include <gridtools/preprocessor/cat.hpp>
# endif
#
# define GT_PP_VARIADIC_HAS_OPT_FUNCTION(...) \
    __VA_OPT__(,) , 1, 0 \
/**/
#
# if GT_PP_VARIADICS_MSVC
# define GT_PP_VARIADIC_HAS_OPT_ELEM0(e0, ...) GT_PP_CAT(GT_PP_VARIADIC_HAS_OPT_ELEM_0(e0,__VA_ARGS__),)
# define GT_PP_VARIADIC_HAS_OPT_ELEM2(e0, ...) GT_PP_CAT(GT_PP_VARIADIC_HAS_OPT_ELEM_2(e0,__VA_ARGS__),)
# else
# define GT_PP_VARIADIC_HAS_OPT_ELEM0(e0, ...) GT_PP_VARIADIC_HAS_OPT_ELEM_0(e0,__VA_ARGS__)
# define GT_PP_VARIADIC_HAS_OPT_ELEM2(e0, ...) GT_PP_VARIADIC_HAS_OPT_ELEM_2(e0,__VA_ARGS__)
# endif
# define GT_PP_VARIADIC_HAS_OPT_ELEM_0(e0, ...) e0
# define GT_PP_VARIADIC_HAS_OPT_ELEM_2(e0, e1, e2, ...) e2
#
# endif
#
# endif
