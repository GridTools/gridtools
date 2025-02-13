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
# ifndef GT_PREPROCESSOR_ITERATION_ITERATE_HPP
# define GT_PREPROCESSOR_ITERATION_ITERATE_HPP
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/arithmetic/inc.hpp>
# include <gridtools/preprocessor/array/elem.hpp>
# include <gridtools/preprocessor/array/size.hpp>
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/slot/slot.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
#
# /* GT_PP_ITERATION_DEPTH */
#
# define GT_PP_ITERATION_DEPTH() 0
#
# /* GT_PP_ITERATION */
#
# define GT_PP_ITERATION() GT_PP_CAT(GT_PP_ITERATION_, GT_PP_ITERATION_DEPTH())
#
# /* GT_PP_ITERATION_START && GT_PP_ITERATION_FINISH */
#
# define GT_PP_ITERATION_START() GT_PP_CAT(GT_PP_ITERATION_START_, GT_PP_ITERATION_DEPTH())
# define GT_PP_ITERATION_FINISH() GT_PP_CAT(GT_PP_ITERATION_FINISH_, GT_PP_ITERATION_DEPTH())
#
# /* GT_PP_ITERATION_FLAGS */
#
# define GT_PP_ITERATION_FLAGS() (GT_PP_CAT(GT_PP_ITERATION_FLAGS_, GT_PP_ITERATION_DEPTH())())
#
# /* GT_PP_FRAME_ITERATION */
#
# define GT_PP_FRAME_ITERATION(i) GT_PP_CAT(GT_PP_ITERATION_, i)
#
# /* GT_PP_FRAME_START && GT_PP_FRAME_FINISH */
#
# define GT_PP_FRAME_START(i) GT_PP_CAT(GT_PP_ITERATION_START_, i)
# define GT_PP_FRAME_FINISH(i) GT_PP_CAT(GT_PP_ITERATION_FINISH_, i)
#
# /* GT_PP_FRAME_FLAGS */
#
# define GT_PP_FRAME_FLAGS(i) (GT_PP_CAT(GT_PP_ITERATION_FLAGS_, i)())
#
# /* GT_PP_RELATIVE_ITERATION */
#
# define GT_PP_RELATIVE_ITERATION(i) GT_PP_CAT(GT_PP_RELATIVE_, i)(GT_PP_ITERATION_)
#
# define GT_PP_RELATIVE_0(m) GT_PP_CAT(m, GT_PP_ITERATION_DEPTH())
# define GT_PP_RELATIVE_1(m) GT_PP_CAT(m, GT_PP_DEC(GT_PP_ITERATION_DEPTH()))
# define GT_PP_RELATIVE_2(m) GT_PP_CAT(m, GT_PP_DEC(GT_PP_DEC(GT_PP_ITERATION_DEPTH())))
# define GT_PP_RELATIVE_3(m) GT_PP_CAT(m, GT_PP_DEC(GT_PP_DEC(GT_PP_DEC(GT_PP_ITERATION_DEPTH()))))
# define GT_PP_RELATIVE_4(m) GT_PP_CAT(m, GT_PP_DEC(GT_PP_DEC(GT_PP_DEC(GT_PP_DEC(GT_PP_ITERATION_DEPTH())))))
#
# /* GT_PP_RELATIVE_START && GT_PP_RELATIVE_FINISH */
#
# define GT_PP_RELATIVE_START(i) GT_PP_CAT(GT_PP_RELATIVE_, i)(GT_PP_ITERATION_START_)
# define GT_PP_RELATIVE_FINISH(i) GT_PP_CAT(GT_PP_RELATIVE_, i)(GT_PP_ITERATION_FINISH_)
#
# /* GT_PP_RELATIVE_FLAGS */
#
# define GT_PP_RELATIVE_FLAGS(i) (GT_PP_CAT(GT_PP_RELATIVE_, i)(GT_PP_ITERATION_FLAGS_)())
#
# /* GT_PP_ITERATE */
#
# define GT_PP_ITERATE() GT_PP_CAT(GT_PP_ITERATE_, GT_PP_INC(GT_PP_ITERATION_DEPTH()))
#
# define GT_PP_ITERATE_1 <gridtools/preprocessor/iteration/detail/iter/forward1.hpp>
# define GT_PP_ITERATE_2 <gridtools/preprocessor/iteration/detail/iter/forward2.hpp>
# define GT_PP_ITERATE_3 <gridtools/preprocessor/iteration/detail/iter/forward3.hpp>
# define GT_PP_ITERATE_4 <gridtools/preprocessor/iteration/detail/iter/forward4.hpp>
# define GT_PP_ITERATE_5 <gridtools/preprocessor/iteration/detail/iter/forward5.hpp>
#
# endif
