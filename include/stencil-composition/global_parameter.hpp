#pragma once

#include "../common/defs.hpp"
#include "../storage/storage.hpp"
#include "../storage/meta_storage.hpp"

namespace gridtools {


/**@brief generic argument type
   struct implementing the minimal interface in order to be passed as an argument to the user functor.
*/
template <typename D, typename MetaData>
struct global_parameter {
    typedef D basic_type;
    typedef D* iterator_type;
    typedef D value_type; //TODO remove
    static const ushort_t field_dimensions=1; //TODO remove

//TODO: This seems to be pretty static. Maybe we should ask
//storage_traits or backend_traits what pointer type to use
#ifdef _USE_GPU_
    typedef hybrid_pointer< D, false > storage_ptr_t;
    typedef hybrid_pointer< const MetaData, false > meta_data_ptr_t;
#else
    typedef wrap_pointer< D, false > storage_ptr_t;
    typedef wrap_pointer< const MetaData, false > meta_data_ptr_t;
#endif
    storage_ptr_t m_storage;
    typedef MetaData storage_info_type;
    meta_data_ptr_t m_meta_data;
    global_parameter(D* this_d, MetaData const& meta_data) : m_storage(this_d, true), m_meta_data(new MetaData(meta_data), false) {}

    GT_FUNCTION
    D *get_pointer_to_use() { return m_storage.get_pointer_to_use(); }

    GT_FUNCTION
    pointer< storage_ptr_t > get_storage_pointer() { return pointer< storage_ptr_t >(&m_storage); }

    GT_FUNCTION
    pointer< const storage_ptr_t > get_storage_pointer() const { return pointer< const storage_ptr_t >(&m_storage); }

    GT_FUNCTION
    pointer< const MetaData > get_meta_data_pointer() const { return pointer< const MetaData >(m_meta_data.get_pointer_to_use()); }

    template<typename ID>
    GT_FUNCTION
    D * access_value() const {return const_cast<D*>(m_storage.get_pointer_to_use());} //TODO change this?

    GT_FUNCTION
    void clone_to_device() {
        m_meta_data.update_gpu();
        m_storage.update_gpu();
    }
};

#define MAKE_GLOBAL_PARAMETER(NAME, FUNCS) \
template <typename MetaData> \
struct NAME;\
namespace gridtools {\
template <typename MetaData> \
struct is_any_storage< NAME<MetaData> > : boost::mpl::true_ {};\
}\
template <typename MetaData> \
struct NAME : global_parameter< NAME<MetaData>, MetaData > {\
    typedef global_parameter<NAME<MetaData>, MetaData> user_defined_storage_t; \
    NAME(MetaData const& meta_data) : user_defined_storage_t(this, meta_data) {} \
    ~NAME() {} \
    FUNCS \
};

} // namespace gridtools
