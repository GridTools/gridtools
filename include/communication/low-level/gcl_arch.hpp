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
#ifndef _GCL_ARCH_
#define _GCL_ARCH_

/** \file
    In this file the different types of architectures are
    defined. The Architectures specifies where the data to be
    exchanged by communication patterns are residing. The possible
    choices are: \link gridtools::gcl_cpu \endlink , \link gridtools::gcl_gpu
    \endlink , end, for illustration purpose only, \link gridtools::gcl_mic
    \endlink , which is not supported at the moment.

    The assumption is that data to be exchanged is in the same place
    for all the processes involved in a pattern. That is, it is not
    possible to send data from a cpu main memory to a remote GPU
    memory.
*/

namespace gridtools {
    /** Indicate that the data is on the main memory of the process
     */
    struct gcl_cpu {};

    /** Indicates that the data is on the memory of a GPU
     */
    struct gcl_gpu {};

    /** Indicates that the data is on the memory of a MIC card.

        Note: Not supported, placed here only for illustration.
     */
    struct gcl_mic {}; // Not supported, placed here only for illustration.
}

#endif
