# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2002.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_REPETITION_DEDUCE_Z_HPP
# define GT_PREPROCESSOR_REPETITION_DEDUCE_Z_HPP
#
# include <gridtools/preprocessor/detail/auto_rec.hpp>
# include <gridtools/preprocessor/repetition/repeat.hpp>
#
# /* GT_PP_DEDUCE_Z */
#
# define GT_PP_DEDUCE_Z() GT_PP_AUTO_REC(GT_PP_REPEAT_P, 4)
#
# endif
