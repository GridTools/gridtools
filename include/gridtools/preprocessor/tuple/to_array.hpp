# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2011.                                    *
#  *     (C) Copyright Paul Mensonides 2011.                                  *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_TUPLE_TO_ARRAY_HPP
# define GT_PREPROCESSOR_TUPLE_TO_ARRAY_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/if.hpp>
# include <gridtools/preprocessor/facilities/overload.hpp>
# include <gridtools/preprocessor/tuple/size.hpp>
# include <gridtools/preprocessor/variadic/size.hpp>
# include <gridtools/preprocessor/variadic/has_opt.hpp>
#
# /* GT_PP_TUPLE_TO_ARRAY */
#
# if GT_PP_VARIADICS_MSVC
#     define GT_PP_TUPLE_TO_ARRAY(...) GT_PP_TUPLE_TO_ARRAY_I(GT_PP_OVERLOAD(GT_PP_TUPLE_TO_ARRAY_, __VA_ARGS__), (__VA_ARGS__))
#     define GT_PP_TUPLE_TO_ARRAY_I(m, args) GT_PP_TUPLE_TO_ARRAY_II(m, args)
#     define GT_PP_TUPLE_TO_ARRAY_II(m, args) GT_PP_CAT(m ## args,)
#     define GT_PP_TUPLE_TO_ARRAY_1(tuple) (GT_PP_TUPLE_SIZE(tuple), tuple)
# else
#     define GT_PP_TUPLE_TO_ARRAY(...) GT_PP_OVERLOAD(GT_PP_TUPLE_TO_ARRAY_, __VA_ARGS__)(__VA_ARGS__)
#     if GT_PP_VARIADIC_HAS_OPT()
#         define GT_PP_TUPLE_TO_ARRAY_1(tuple) GT_PP_TUPLE_TO_ARRAY_1_SIZE(GT_PP_VARIADIC_SIZE tuple, tuple)
#         define GT_PP_TUPLE_TO_ARRAY_1_SIZE(size,tuple) (GT_PP_IF(size,size,1), tuple)
#     else
#         define GT_PP_TUPLE_TO_ARRAY_1(tuple) (GT_PP_VARIADIC_SIZE tuple, tuple)
#     endif
# endif
# define GT_PP_TUPLE_TO_ARRAY_2(size, tuple) (size, tuple)
#
# endif
