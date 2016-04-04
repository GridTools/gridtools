#pragma once

/**@file
   @brief contains the API of the caonditionals type, to be used for specifying the control flow
   in the computation tree.

   The user wanting to select a multi-stage stencil at runtime, based on a boolean condition, must instantiate
   this class with a unique ID as template argument, construct it using the boolean condition, and then
   use the \ref gridtools::if_ statement from whithin the make_computation.
*/
#ifdef CXX11_ENABLED
#define BOOL_FUNC(val) std::function< bool() > val
#else
#define BOOL_FUNC(val) bool (*val)()
#endif

namespace gridtools {

    template < uint_t Tag, uint_t SwitchId = 0 >
    class conditional {

        // weak pointer, viewing the boolean condition
        BOOL_FUNC(m_value);

      public:
        typedef static_uint< Tag > index_t;
        static const uint_t index_value = index_t::value;

        /**
           @brief default constructor
         */
        conditional() // try to avoid this?
            : m_value(
#ifdef CXX11_ENABLED
                  []() {
                      assert(false);
                      return false;
                  }
#endif
                  ) {
        }

        /**
           @brief constructor from a pointer
         */
        conditional(BOOL_FUNC(c)) : m_value(c) {}

        /**@brief returns the boolean condition*/
        bool value() const { return m_value(); }
    };

    template < typename T >
    struct is_conditional : boost::mpl::false_ {};

    template < uint_t Tag >
    struct is_conditional< conditional< Tag > > : boost::mpl::true_ {};

    template < uint_t Tag, uint_t SwitchId >
    struct is_conditional< conditional< Tag, SwitchId > > : boost::mpl::true_ {};
}
