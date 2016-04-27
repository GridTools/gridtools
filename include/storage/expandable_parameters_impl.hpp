#pragma once

// namespace gridtools{
//     namespace _impl{

//         template <typename Pointer>
//         struct assign_storage_pointer_impl{
//         private:
//             Pointer& m_pointer;

//         public:
//             assign_storage_pointer_impl(Pointer& pointer_) :
//                 m_pointer(pointer_){}

//             template<typename Storage>
//             void operator = (Storage & storage_){
//                 GRIDTOOLS_STATIC_ASSERT(Storage::field_dimensions==1,
//                                         "The storage on the right hand side has field dimension larger than one. you cannot assign an element in an expandable parameter list to another list of storage.");
//                 //assigning wrap (or hybrid) pointer
//                 m_pointer = storage_.fields_view()[0];
//             }
//         };
//     } //namespace _impl
// } //namespace gridtools
