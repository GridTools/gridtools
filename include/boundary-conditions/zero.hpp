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

/**
   @file
   @brief On all boundary the values ares set to DataField::value_type(), which is zero for basic data types.
*/

namespace gridtools {

    /**
       @brief On all boundary the values ares set to DataField::value_type(), which is zero for basic data types.
    */
    struct zero_boundary {

        template < typename Direction, typename DataField0 >
        GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = typename DataField0::value_type();
        }

        template < typename Direction, typename DataField0, typename DataField1 >
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = typename DataField0::value_type();
            data_field1(i, j, k) = typename DataField1::value_type();
        }

        template < typename Direction, typename DataField0, typename DataField1, typename DataField2 >
        GT_FUNCTION void operator()(Direction,
            DataField0 &data_field0,
            DataField1 &data_field1,
            DataField2 &data_field2,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field0(i, j, k) = typename DataField0::value_type();
            data_field1(i, j, k) = typename DataField1::value_type();
            data_field2(i, j, k) = typename DataField2::value_type();
        }
    };

} // namespace gridtools
