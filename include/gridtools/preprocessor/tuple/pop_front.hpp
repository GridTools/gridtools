# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2013.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_TUPLE_POP_FRONT_HPP
# define GT_PREPROCESSOR_TUPLE_POP_FRONT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/array/pop_front.hpp>
# include <gridtools/preprocessor/array/to_tuple.hpp>
# include <gridtools/preprocessor/comparison/greater.hpp>
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/tuple/size.hpp>
# include <gridtools/preprocessor/tuple/to_array.hpp>
#
#
# /* GT_PP_TUPLE_POP_FRONT */
#
# define GT_PP_TUPLE_POP_FRONT(tuple) \
    GT_PP_IIF \
        ( \
        GT_PP_GREATER(GT_PP_TUPLE_SIZE(tuple),1), \
        GT_PP_TUPLE_POP_FRONT_EXEC, \
        GT_PP_TUPLE_POP_FRONT_RETURN \
        ) \
    (tuple) \
/**/
#
# define GT_PP_TUPLE_POP_FRONT_EXEC(tuple) \
    GT_PP_ARRAY_TO_TUPLE(GT_PP_ARRAY_POP_FRONT(GT_PP_TUPLE_TO_ARRAY(tuple))) \
/**/
#
# define GT_PP_TUPLE_POP_FRONT_RETURN(tuple) tuple
#
# /* GT_PP_TUPLE_POP_FRONT_Z */
#
# define GT_PP_TUPLE_POP_FRONT_Z(z, tuple) \
    GT_PP_IIF \
        ( \
        GT_PP_GREATER(GT_PP_TUPLE_SIZE(tuple),1), \
        GT_PP_TUPLE_POP_FRONT_Z_EXEC, \
        GT_PP_TUPLE_POP_FRONT_Z_RETURN \
        ) \
    (z, tuple) \
/**/
#
# define GT_PP_TUPLE_POP_FRONT_Z_EXEC(z, tuple) \
    GT_PP_ARRAY_TO_TUPLE(GT_PP_ARRAY_POP_FRONT_Z(z, GT_PP_TUPLE_TO_ARRAY(tuple))) \
/**/
#
# define GT_PP_TUPLE_POP_FRONT_Z_RETURN(z, tuple) tuple
#
# endif // GT_PREPROCESSOR_TUPLE_POP_FRONT_HPP
