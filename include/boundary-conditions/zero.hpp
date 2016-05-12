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

        template <typename Direction, typename DataField0>
        GT_FUNCTION
        void operator()(Direction,
                        DataField0 & data_field0,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0(i,j,k) = typename DataField0::value_type();
        }

        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION
        void operator()(Direction,
                        DataField0 & data_field0,
                        DataField1 & data_field1,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0(i,j,k) = typename DataField0::value_type();
            data_field1(i,j,k) = typename DataField1::value_type();
        }

        template <typename Direction, typename DataField0, typename DataField1, typename DataField2>
        GT_FUNCTION
        void operator()(Direction,
                        DataField0 & data_field0,
                        DataField1 & data_field1,
                        DataField2 & data_field2,
                        uint_t i, uint_t j, uint_t k) const {
            data_field0(i,j,k) = typename DataField0::value_type();
            data_field1(i,j,k) = typename DataField1::value_type();
            data_field2(i,j,k) = typename DataField2::value_type();
        }
    };



} // namespace gridtools
