# Try to find clang-tidy and clang-format
#
# Usage of this module:
#
#  find_package(ClangTools)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#  ClangTools_PATH      When set, this path is inspected instead of standard library binary 
#                       locations to find clang-tidy and clang-format
#
# This module defines:
#  CLANG_TIDY_BIN       The  path to the clang tidy binary
#  CLANG_TIDY_FOUND     Whether clang tidy was found
#  CLANG_FORMAT_BIN     The path to the clang format binary 
#  CLANG_FORMAT_FOUND   Whether clang format was found
#

find_program(CLANG_TIDY_BIN 
  NAMES clang-tidy-3.7.1 clang-tidy
  PATHS ${ClangTools_PATH} $ENV{CLANG_TOOLS_PATH} ~/bin /usr/local/bin /usr/bin 
        NO_DEFAULT_PATH
)

if ("${CLANG_TIDY_BIN}" STREQUAL "CLANG_TIDY_BIN-NOTFOUND") 
  set(CLANG_TIDY_FOUND 0)
  message(STATUS "clang-tidy not found")
else()
  set(CLANG_TIDY_FOUND 1)
  message(STATUS "clang-tidy found at ${CLANG_TIDY_BIN}")
endif()

find_program(CLANG_FORMAT_BIN 
  NAMES clang-format-3.7.1 clang-format
  PATHS ${ClangTools_PATH} $ENV{CLANG_TOOLS_PATH}  ~/bin /usr/local/bin /usr/bin 
        NO_DEFAULT_PATH
)

if("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND") 
  set(CLANG_FORMAT_FOUND 0)
  message(STATUS "clang-format not found")
else()
  set(CLANG_FORMAT_FOUND 1)
  message(STATUS "clang-format found at ${CLANG_FORMAT_BIN}")
endif()
