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
#ifndef _DESCRIPTORS_FWD_H_
#define _DESCRIPTORS_FWD_H_

namespace gridtools {
    template < typename DataType, int DIMS, typename >
    class hndlr_descriptor_ut;

    template < typename Datatype, typename GridType, typename, typename, typename, int >
    class hndlr_dynamic_ut;

    template < int DIMS,
        typename Haloexch,
        typename proc_layout_abs = typename default_layout_map< DIMS >::type,
        typename Gcl_Arch = gcl_cpu,
        int = version_mpi_pack >
    class hndlr_generic;

    template < typename DataType, typename layoutmap, template < typename > class traits >
    struct field_on_the_fly;

    template < int DIMS, typename Haloexch, typename proc_layout, typename Gcl_Arch, int versiono >
    class hndlr_generic;
}
#endif
