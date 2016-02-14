#pragma once

/**@file
   @brief contains the API of the caonditionals type, to be used for specifying the control flow
   in the computation tree.

   The user wanting to select a multi-stage stencil at runtime, based on a boolean condition, must instantiate
   this class with a unique ID as template argument, construct it using the boolean condition, and then
   use the \ref gridtools::if_ statement from whithin the make_computation.
*/

namespace gridtools{

/**   @brief contains the API of the caonditionals type, to be used for specifying the control flow
      in the computation tree.

      The user wanting to select a multi-stage stencil at runtime, based on a boolean condition, must instantiate
      this class with a unique ID as template argument, construct it using the boolean condition, and then
      use the \ref gridtools::if_ statement from whithin the make_computation.
*/
    template <uint_t Tag>
    class conditional{

        //weak pointer
        short_t* m_value;

    public:
        typedef static_uint<Tag> index_t;
        static const uint_t index_value = index_t::value;

        /**
           @brief default constructor
         */
        constexpr conditional () //try to avoid this?
            : m_value()
        {}

        /**
           @brief constructor from a pointer
         */
        constexpr conditional (short_t*& c)
            : m_value(c)
        {}

        /**
           @brief constructor from a reference value
         */
        constexpr conditional (short_t& c)
            : m_value(&c)
        {}

        /**
           @brief copy constructor

           makes a shallow copy of the pointer: this class does not
           have ownership of the condition, it is just a viewer, the owner being either
           the user, or the switch_variable if the conditionals are being used by the library to
           implement a switch.
         */
        constexpr conditional (conditional const& other)
            : m_value(other.m_value)//shallow copy the pointer
        {}

        /**@brief returns the pointer*/
        constexpr short_t* value_ptr() const {return m_value;}

        /**@brief returns the boolean condition*/
        constexpr bool value() const {return *m_value;}
    };

    template <typename T>
    struct is_conditional : boost::mpl::false_ {};

    template <uint_t Tag>
    struct is_conditional<conditional<Tag> >:boost::mpl::true_ {};
}
