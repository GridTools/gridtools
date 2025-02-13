# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002) */
# /* Revised by Edward Diener (2011,2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_CONFIG_LIMITS_HPP
# define GT_PREPROCESSOR_CONFIG_LIMITS_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if defined(GT_PP_LIMIT_DIM)
# undef GT_PP_LIMIT_DIM
# endif
# if defined(GT_PP_LIMIT_ITERATION_DIM)
# undef GT_PP_LIMIT_ITERATION_DIM
# endif
# if defined(GT_PP_LIMIT_SLOT_SIG)
# undef GT_PP_LIMIT_SLOT_SIG
# endif
# if defined(GT_PP_LIMIT_SLOT_COUNT)
# undef GT_PP_LIMIT_SLOT_COUNT
# endif
# if defined(GT_PP_LIMIT_WHILE)
# undef GT_PP_LIMIT_WHILE
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# if defined(GT_PP_LIMIT_MAG)
# undef GT_PP_LIMIT_MAG
# endif
# if defined(GT_PP_LIMIT_VARIADIC)
# undef GT_PP_LIMIT_VARIADIC
# endif
# if defined(GT_PP_LIMIT_TUPLE)
# undef GT_PP_LIMIT_TUPLE
# endif
# if defined(GT_PP_LIMIT_FOR)
# undef GT_PP_LIMIT_FOR
# endif
# if defined(GT_PP_LIMIT_REPEAT)
# undef GT_PP_LIMIT_REPEAT
# endif
# if defined(GT_PP_LIMIT_SEQ)
# undef GT_PP_LIMIT_SEQ
# endif
# if defined(GT_PP_LIMIT_ITERATION)
# undef GT_PP_LIMIT_ITERATION
# endif
#
# define GT_PP_LIMIT_MAG 256
# define GT_PP_LIMIT_WHILE 256
# define GT_PP_LIMIT_VARIADIC 64
# define GT_PP_LIMIT_TUPLE 64
# define GT_PP_LIMIT_FOR 256
# define GT_PP_LIMIT_SEQ 256
# define GT_PP_LIMIT_REPEAT 256
# define GT_PP_LIMIT_ITERATION 256
#
#else
#
# if defined(GT_PP_LIMIT_MAG)
# if !(GT_PP_LIMIT_MAG == 256 || GT_PP_LIMIT_MAG == 512 || GT_PP_LIMIT_MAG == 1024)
# undef GT_PP_LIMIT_MAG
# define GT_PP_LIMIT_MAG 256
# define GT_PP_LIMIT_WHILE 256
# else
# define GT_PP_LIMIT_WHILE GT_PP_LIMIT_MAG
# if !defined(GT_PP_LIMIT_SEQ)
# define GT_PP_LIMIT_SEQ GT_PP_LIMIT_MAG
# endif
# endif
# else
# define GT_PP_LIMIT_MAG 256
# define GT_PP_LIMIT_WHILE 256
# endif
#
# if defined(GT_PP_LIMIT_VARIADIC)
# if !(GT_PP_LIMIT_VARIADIC == 64 || GT_PP_LIMIT_VARIADIC == 128 || GT_PP_LIMIT_VARIADIC == 256)
# undef GT_PP_LIMIT_VARIADIC
# define GT_PP_LIMIT_VARIADIC 64
# endif
# else
# define GT_PP_LIMIT_VARIADIC 64
# endif
#
# if defined(GT_PP_LIMIT_TUPLE)
# if !(GT_PP_LIMIT_TUPLE == 64 || GT_PP_LIMIT_TUPLE == 128 || GT_PP_LIMIT_TUPLE == 256)
# undef GT_PP_LIMIT_TUPLE
# define GT_PP_LIMIT_TUPLE 64
# elif GT_PP_LIMIT_TUPLE > GT_PP_LIMIT_VARIADIC
# undef GT_PP_LIMIT_VARIADIC
# define GT_PP_LIMIT_VARIADIC GT_PP_LIMIT_TUPLE
# endif
# else
# define GT_PP_LIMIT_TUPLE 64
# endif
#
# if defined(GT_PP_LIMIT_FOR)
# if !(GT_PP_LIMIT_FOR == 256 || GT_PP_LIMIT_FOR == 512 || GT_PP_LIMIT_FOR == 1024)
# undef GT_PP_LIMIT_FOR
# define GT_PP_LIMIT_FOR 256
# elif GT_PP_LIMIT_FOR > GT_PP_LIMIT_MAG
# undef GT_PP_LIMIT_FOR
# define GT_PP_LIMIT_FOR GT_PP_LIMIT_MAG
# endif
# else
# define GT_PP_LIMIT_FOR 256
# endif
#
# if defined(GT_PP_LIMIT_REPEAT)
# if !(GT_PP_LIMIT_REPEAT == 256 || GT_PP_LIMIT_REPEAT == 512 || GT_PP_LIMIT_REPEAT == 1024)
# undef GT_PP_LIMIT_REPEAT
# define GT_PP_LIMIT_REPEAT 256
# elif GT_PP_LIMIT_REPEAT > GT_PP_LIMIT_MAG
# undef GT_PP_LIMIT_REPEAT
# define GT_PP_LIMIT_REPEAT GT_PP_LIMIT_MAG
# endif
# else
# define GT_PP_LIMIT_REPEAT 256
# endif
#
# if defined(GT_PP_LIMIT_SEQ)
# if !(GT_PP_LIMIT_SEQ == 256 || GT_PP_LIMIT_SEQ == 512 || GT_PP_LIMIT_SEQ == 1024)
# undef GT_PP_LIMIT_SEQ
# define GT_PP_LIMIT_SEQ 256
# elif GT_PP_LIMIT_SEQ > GT_PP_LIMIT_MAG
# undef GT_PP_LIMIT_SEQ
# define GT_PP_LIMIT_SEQ GT_PP_LIMIT_MAG
# endif
# else
# define GT_PP_LIMIT_SEQ 256
# endif
#
# if defined(GT_PP_LIMIT_ITERATION)
# if !(GT_PP_LIMIT_ITERATION == 256 || GT_PP_LIMIT_ITERATION == 512 || GT_PP_LIMIT_ITERATION == 1024)
# undef GT_PP_LIMIT_ITERATION
# define GT_PP_LIMIT_ITERATION 256
# elif GT_PP_LIMIT_ITERATION > GT_PP_LIMIT_MAG
# undef GT_PP_LIMIT_ITERATION
# define GT_PP_LIMIT_ITERATION GT_PP_LIMIT_MAG
# endif
# else
# define GT_PP_LIMIT_ITERATION 256
# endif
#
# endif
#
# define GT_PP_LIMIT_DIM 3
# define GT_PP_LIMIT_ITERATION_DIM 3
# define GT_PP_LIMIT_SLOT_SIG 10
# define GT_PP_LIMIT_SLOT_COUNT 5
#
# endif
