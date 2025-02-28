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
# ifndef GT_PREPROCESSOR_SLOT_SLOT_HPP
# define GT_PREPROCESSOR_SLOT_SLOT_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/slot/detail/def.hpp>
#
# /* GT_PP_ASSIGN_SLOT */
#
# define GT_PP_ASSIGN_SLOT(i) GT_PP_CAT(GT_PP_ASSIGN_SLOT_, i)
#
# define GT_PP_ASSIGN_SLOT_1 <gridtools/preprocessor/slot/detail/slot1.hpp>
# define GT_PP_ASSIGN_SLOT_2 <gridtools/preprocessor/slot/detail/slot2.hpp>
# define GT_PP_ASSIGN_SLOT_3 <gridtools/preprocessor/slot/detail/slot3.hpp>
# define GT_PP_ASSIGN_SLOT_4 <gridtools/preprocessor/slot/detail/slot4.hpp>
# define GT_PP_ASSIGN_SLOT_5 <gridtools/preprocessor/slot/detail/slot5.hpp>
#
# /* GT_PP_SLOT */
#
# define GT_PP_SLOT(i) GT_PP_CAT(GT_PP_SLOT_, i)()
#
# endif
