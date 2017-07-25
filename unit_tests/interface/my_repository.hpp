#include <unordered_map>
#include "storage/storage-facility.hpp"
#include "boost/variant.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

// namespace {
//    template < typename T >
//    struct type_to_id {
//        GRIDTOOLS_STATIC_ASSERT(sizeof(T) < 0, GT_INTERNAL_ERROR);
//    };
//    template < typename T >
//    struct id_to_type {
//        GRIDTOOLS_STATIC_ASSERT(sizeof(T) < 0, GT_INTERNAL_ERROR);
//    };
//#define GT_REGISTER_FIELDTYPE(TYPE)                      \
//    static const int uniqueid_##TYPE = __COUNTER__;      \
//    template <>                                          \
//    struct type_to_id< TYPE > {                          \
//        const static int value = uniqueid_##TYPE;        \
//    };                                                   \
//    template <>                                          \
//    struct id_to_type< static_int< uniqueid_##TYPE > > { \
//        using type = TYPE;                               \
//    };
//#define GT_REGISTER_FIELD(TYPE, NAME)
//#include "my_repository.inc"
//#undef GT_REGISTER_FIELD
//#undef GT_REGISTER_FIELDTYPE
//}

#define FEL1(Action, X) Action(X)
#define FEL2(Action, X, ...) Action(X), FEL1(Action, __VA_ARGS__)
#define FEL3(Action, X, ...) Action(X), FEL2(Action, __VA_ARGS__)
#define FEL4(Action, X, ...) Action(X), FEL3(Action, __VA_ARGS__)
#define FEL5(Action, X, ...) Action(X), FEL4(Action, __VA_ARGS__)
#define FEL6(Action, X, ...) Action(X), FEL5(Action, __VA_ARGS__)
#define FEL7(Action, X, ...) Action(X), FEL6(Action, __VA_ARGS__)
#define FEL8(Action, X, ...) Action(X), FEL7(Action, __VA_ARGS__)
#define FEL9(Action, X, ...) Action(X), FEL8(Action, __VA_ARGS__)
#define FEL10(Action, X, ...) Action(X), FEL9(Action, __VA_ARGS__)
#define FEL11(Action, X, ...) Action(X), FEL10(Action, __VA_ARGS__)
#define FEL12(Action, X, ...) Action(X), FEL11(Action, __VA_ARGS__)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, Name, ...) Name
#define FOR_EACH_LIST(Action, ...) \
    GET_MACRO(__VA_ARGS__, FEL12, FEL11, FEL10, FEL9, FEL8, FEL7, FEL6, FEL5, FEL4, FEL3, FEL2, FEL1)(Action,__VA_ARGS__)

class my_repository {
  private:
#define GT_REGISTER_FIELD(Type, Name) Type m_##Name;
#define GT_REGISTER_FIELDTYPES(...)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
    int dummy;

  public:
    std::unordered_map< std::string,
        boost::variant<
#define JustForward(X) X
#define GT_REGISTER_FIELD(Type, Name)
#define GT_REGISTER_FIELDTYPES(...) FOR_EACH_LIST(JustForward, __VA_ARGS__)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
                            > > data_store_map_; // TODO fix the comma

#define GET_STORAGE_INFO(X) typename X::storage_info_t info_##X
    my_repository(
#define MakeCtor(Type) typename Type::storage_info_t info_##Type
#define GT_REGISTER_FIELD(Type, Name)
#define GT_REGISTER_FIELDTYPES(...) FOR_EACH_LIST(MakeCtor, __VA_ARGS__)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
        )
        :
#define GT_REGISTER_FIELD(Type, Name) m_##Name(info_##Type),
#define GT_REGISTER_FIELDTYPES(...)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
          dummy(0) { // TODO FIXME
#define GT_REGISTER_FIELD(Type, Name) data_store_map_.emplace(#Name, m_##Name);
#define GT_REGISTER_FIELDTYPES(...)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES
    }

    /// @brief Configuration is non-copyable/moveable
    my_repository(const my_repository &) = delete;
    my_repository(my_repository &&) = delete;

// Getter & Setter
#define GT_REGISTER_FIELD(Type, Name)                                       \
    const Type &get_##Name() const noexcept { return this->m_##Name; }      \
    void set_##Name(const Type &value) noexcept { this->m_##Name = value; } \
    void init_##Name(const Type::storage_info_t &storage_info) { this->m_##Name.allocate(storage_info); }
#define GT_REGISTER_FIELDTYPES(...)
#include "my_repository.inc"
#undef GT_REGISTER_FIELD
#undef GT_REGISTER_FIELDTYPES

    //#define GT_REGISTER_FIELD(Type, Name)
    //#define GT_REGISTER_FIELDTYPE(Type) \
//    const std::vector< std::reference_wrapper< Type > > &get_##Type() const noexcept { return this->m_map_##Type; }
    //#include "my_repository.inc"
    //#undef GT_REGISTER_FIELD
    //#undef GT_REGISTER_FIELDTYPE
};
