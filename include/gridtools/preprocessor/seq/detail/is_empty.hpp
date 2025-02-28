# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2015.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_SEQ_DETAIL_IS_EMPTY_HPP
# define GT_PREPROCESSOR_SEQ_DETAIL_IS_EMPTY_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/logical/bool.hpp>
# include <gridtools/preprocessor/logical/compl.hpp>
# include <gridtools/preprocessor/seq/size.hpp>
#
/* An empty seq is one that is just GT_PP_SEQ_NIL */
#
# define GT_PP_SEQ_DETAIL_IS_EMPTY(seq) \
    GT_PP_COMPL \
        ( \
        GT_PP_SEQ_DETAIL_IS_NOT_EMPTY(seq) \
        ) \
/**/
#
# define GT_PP_SEQ_DETAIL_IS_EMPTY_SIZE(size) \
    GT_PP_COMPL \
        ( \
        GT_PP_SEQ_DETAIL_IS_NOT_EMPTY_SIZE(size) \
        ) \
/**/
#
# define GT_PP_SEQ_DETAIL_IS_NOT_EMPTY(seq) \
    GT_PP_SEQ_DETAIL_IS_NOT_EMPTY_SIZE(GT_PP_SEQ_DETAIL_EMPTY_SIZE(seq)) \
/**/
#
# define GT_PP_SEQ_DETAIL_IS_NOT_EMPTY_SIZE(size) \
    GT_PP_BOOL(size) \
/**/
#
# define GT_PP_SEQ_DETAIL_EMPTY_SIZE(seq) \
    GT_PP_DEC(GT_PP_SEQ_SIZE(seq (nil))) \
/**/
#
# endif
