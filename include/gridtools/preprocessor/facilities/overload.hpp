# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2011.                                  *
#  *     (C) Copyright Edward Diener 2011.                                    *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_FACILITIES_OVERLOAD_HPP
# define GT_PREPROCESSOR_FACILITIES_OVERLOAD_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/variadic/size.hpp>
#
# /* GT_PP_OVERLOAD */
#
# define GT_PP_OVERLOAD(prefix, ...) GT_PP_CAT(prefix, GT_PP_VARIADIC_SIZE(__VA_ARGS__))
#
# endif
