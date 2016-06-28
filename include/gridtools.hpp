/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <boost/config.hpp>
#if defined(__CUDACC__) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include <boost/mpl/greater.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/make_vector.hpp>
#ifndef __CUDACC__
#include <boost/timer/timer.hpp>
#endif
#include <boost/mpl/contains.hpp>
#include <boost/utility/enable_if.hpp>

#include <communication/GCL.hpp>

#include "common/defs.hpp"
#include "common/host_device.hpp"
#include "common/array.hpp"
#include "common/layout_map.hpp"
#include "common/pointer.hpp"
#include "common/pointer_metafunctions.hpp"
