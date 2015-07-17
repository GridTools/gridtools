/**@file
   @brief file with classes to store the data members of the iterate domain
   that will be allocated in shared memory
 */
#pragma once

/**
 * @class shared_iterate_domain
 * data structure that holds data members of the iterate domain that must be stored in shared memory.
 * @tparam
 */
template<typename DataPointerArray, typename StridesType>
class shared_iterate_domain
{
    DISALLOW_COPY_AND_ASSIGN(shared_iterate_domain);
private:
    DataPointerArray m_data_pointer;
    StridesType m_strides;

public:
    shared_iterate_domain(){}

    DataPointerArray const & data_pointer() const { return m_data_pointer;}
    StridesType const & strides() const { return m_strides;}
    DataPointerArray & data_pointer() { return m_data_pointer;}
    StridesType & strides() { return m_strides;}

};
