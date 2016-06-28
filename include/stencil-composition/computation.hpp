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
// //\todo this struct becomes redundant when the auto keyword is used
namespace gridtools {
    template < typename ReductionType = int >
    struct computation {
        virtual void ready() = 0;
        virtual void steady() = 0;
        virtual void finalize() = 0;
        virtual ReductionType run() = 0;
        virtual std::string print_meter() = 0;
        virtual double get_meter() = 0;
    };

} // namespace gridtools
