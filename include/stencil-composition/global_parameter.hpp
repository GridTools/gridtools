#pragma once

#include "../common/defs.hpp"
#include "../storage/storage.hpp"
#include "../storage/meta_storage.hpp"

namespace gridtools {


/**@brief generic argument type
   struct implementing the minimal interface in order to be passed as an argument to the user functor.
*/
template <typename T>
struct global_parameter : T {
    // following typedefs are needed to keep compatibility
    // when passing global_parameter as a "storage" to the
    // intermediate.
    typedef global_parameter<T> this_type;
    typedef T wrapped_type;
    typedef this_type basic_type;
    typedef this_type* iterator_type;
    typedef this_type value_type;
    static const ushort_t field_dimensions=1;
    struct storage_info_type {
       typedef void index_type;
    };

//TODO: This seems to be pretty static. Maybe we should ask
//storage_traits or backend_traits what pointer type to use
#ifdef _USE_GPU_
    typedef hybrid_pointer< this_type, false > storage_ptr_t;
#else
    typedef wrap_pointer< this_type, false > storage_ptr_t;
#endif
    storage_ptr_t m_storage;

    global_parameter(T& t) : T(t), m_storage(static_cast<this_type*>(this), true) {}

    GT_FUNCTION
    this_type *get_pointer_to_use() { return m_storage.get_pointer_to_use(); }

    GT_FUNCTION
    pointer< storage_ptr_t > get_storage_pointer() { return pointer< storage_ptr_t >(&m_storage); }

    GT_FUNCTION
    pointer< const storage_ptr_t > get_storage_pointer() const { return pointer< const storage_ptr_t >(&m_storage); }

    template<typename ID>
    GT_FUNCTION
    this_type * access_value() const {return const_cast<this_type*>(m_storage.get_pointer_to_use());} //TODO change this?

    GT_FUNCTION
    void clone_to_device() {
        m_storage.update_gpu();
    }
};

} // namespace gridtools
