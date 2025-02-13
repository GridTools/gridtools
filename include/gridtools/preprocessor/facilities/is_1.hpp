# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2003.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_FACILITIES_IS_1_HPP
# define GT_PREPROCESSOR_FACILITIES_IS_1_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/facilities/is_empty.hpp>
#
# /* GT_PP_IS_1 */
#
# define GT_PP_IS_1(x) GT_PP_IS_EMPTY(GT_PP_CAT(GT_PP_IS_1_HELPER_, x))
# define GT_PP_IS_1_HELPER_1
#
# endif
