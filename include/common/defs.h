#pragma once
/**
@file
@brief global definitions
*/

#define GT_MAX_ARGS 8
#define GT_MAX_INDEPENDENT 3


#define GT_NO_ERRORS 0
#define GT_ERROR_NO_TEMPS 1

namespace gridtools{  namespace enumtype{
/** enum specifying the type of backend we use */
        enum backend  {Cuda, Host};
}}
