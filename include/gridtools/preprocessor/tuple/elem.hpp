# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002-2011) */
# /* Revised by Edward Diener (2011,2014,2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_TUPLE_ELEM_HPP
# define GT_PREPROCESSOR_TUPLE_ELEM_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/facilities/expand.hpp>
# include <gridtools/preprocessor/facilities/overload.hpp>
# include <gridtools/preprocessor/tuple/rem.hpp>
# include <gridtools/preprocessor/variadic/elem.hpp>
# include <gridtools/preprocessor/tuple/detail/is_single_return.hpp>
#
# if GT_PP_VARIADICS_MSVC
#     define GT_PP_TUPLE_ELEM(...) GT_PP_TUPLE_ELEM_I(GT_PP_OVERLOAD(GT_PP_TUPLE_ELEM_O_, __VA_ARGS__), (__VA_ARGS__))
#     define GT_PP_TUPLE_ELEM_I(m, args) GT_PP_TUPLE_ELEM_II(m, args)
#     define GT_PP_TUPLE_ELEM_II(m, args) GT_PP_CAT(m ## args,)
/*
  Use GT_PP_REM_CAT if it is a single element tuple ( which might be empty )
  else use GT_PP_REM. This fixes a VC++ problem with an empty tuple and GT_PP_TUPLE_ELEM
  functionality. See tuple_elem_bug_test.cxx.
*/
#     define GT_PP_TUPLE_ELEM_O_2(n, tuple) \
         GT_PP_VARIADIC_ELEM(n, GT_PP_EXPAND(GT_PP_TUPLE_IS_SINGLE_RETURN(GT_PP_REM_CAT,GT_PP_REM,tuple) tuple)) \
         /**/
# else
#     define GT_PP_TUPLE_ELEM(...) GT_PP_OVERLOAD(GT_PP_TUPLE_ELEM_O_, __VA_ARGS__)(__VA_ARGS__)
#     define GT_PP_TUPLE_ELEM_O_2(n, tuple) GT_PP_VARIADIC_ELEM(n, GT_PP_REM tuple)
# endif
# define GT_PP_TUPLE_ELEM_O_3(size, n, tuple) GT_PP_TUPLE_ELEM_O_2(n, tuple)
#
# /* directly used elsewhere in Boost... */
#
# define GT_PP_TUPLE_ELEM_1_0(a) a
#
# define GT_PP_TUPLE_ELEM_2_0(a, b) a
# define GT_PP_TUPLE_ELEM_2_1(a, b) b
#
# define GT_PP_TUPLE_ELEM_3_0(a, b, c) a
# define GT_PP_TUPLE_ELEM_3_1(a, b, c) b
# define GT_PP_TUPLE_ELEM_3_2(a, b, c) c
#
# endif
