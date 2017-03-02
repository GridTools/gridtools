#pragma once
#ifdef __CUDACC__
#ifndef _GCL_GPU_
#define _GCL_GPU_
#endif
#else
#ifdef _GCL_GPU_
#undef _GCL_GPU_
#endif
#endif
