#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif
#include <storage.h>
#include <cuda_storage.h>
#include <array.h>
#include <layout_map.h>
#include <axis.h>
#include <make_stencils.h>
#include <arg_type.h>
#include <execution_types.h>
#include <domain_type.h>
#include <intermediate.h>

