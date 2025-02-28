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
# ifndef GT_PREPROCESSOR_ARRAY_TO_TUPLE_HPP
# define GT_PREPROCESSOR_ARRAY_TO_TUPLE_HPP
#
# include <gridtools/preprocessor/array/data.hpp>
# include <gridtools/preprocessor/array/size.hpp>
# include <gridtools/preprocessor/control/if.hpp>
#
# /* GT_PP_ARRAY_TO_TUPLE */
#
#    define GT_PP_ARRAY_TO_TUPLE(array) \
        GT_PP_IF \
            ( \
            GT_PP_ARRAY_SIZE(array), \
            GT_PP_ARRAY_DATA, \
            GT_PP_ARRAY_TO_TUPLE_EMPTY \
            ) \
        (array) \
/**/
#    define GT_PP_ARRAY_TO_TUPLE_EMPTY(array)
#
# endif
