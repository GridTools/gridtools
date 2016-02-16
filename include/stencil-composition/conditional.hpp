#pragma once

/**@file
   @brief contains the API of the caonditionals type, to be used for specifying the control flow
   in the computation tree.

   The user wanting to select a multi-stage stencil at runtime, based on a boolean condition, must instantiate
   this class with a unique ID as template argument, construct it using the boolean condition, and then
   use the \ref gridtools::if_ statement from whithin the make_computation.
*/

namespace gridtools{

    template <uint_t Tag>
    class conditional{

        //weak pointer, viewing the boolean condition
        bool m_value;

    public:
        typedef static_uint<Tag> index_t;
        static const uint_t index_value = index_t::value;

        /**
           @brief default constructor
         */
        conditional () //try to avoid this?
            : m_value(false)
        {}

        /**
           @brief constructor from a pointer
         */
        conditional (bool c)
            : m_value(c)
        {}

        /**@brief returns the boolean condition*/
        bool value() const {
            return m_value;
        }
    };


/**   @brief contains the definition of the conditional type used by the switch statement.

      This class is not used directly by the user.
      It is useful only to implement the switch_ statement. The main difference is that when creating a
      switch_ the conditionals get created on the fly, as temporary objects, and thus they cannot be
      owners of the boolean values (which are instead stored in the @ref gridtools::switch_variable,
      which is constructed by the user). Thus in this case the address of the conditional is stored
      in a pointer, accessed and modified but not constructed/descructed by this class.
*/
    template <uint_t Tag, uint_t SwitchId>
    class switch_conditional{

        //weak pointer, viewing the boolean condition
        short_t* m_value;

    public:
        typedef static_uint<Tag> index_t;
        typedef static_uint<SwitchId> switch_id_t;
        static const uint_t index_value = index_t::value;
        static const uint_t switch_id_value = switch_id_t::value;

        /**
           @brief default constructor
         */
        switch_conditional () //try to avoid this?
            : m_value()
        {
        }

        /**
           @brief constructor from a pointer
         */
        switch_conditional (short_t*& c)
            : m_value(c)
        {
        }

        /**
           @brief constructor from a reference value
         */
        switch_conditional (short_t& c)
            : m_value(&c)
        {
        }

        /**
           @brief copy constructor

           makes a shallow copy of the pointer: this class does not
           have ownership of the condition, it is just a viewer, the owner being either
           the user, or the switch_variable if the conditionals are being used by the library to
           implement a switch.
         */
        switch_conditional (switch_conditional const& other)
            : m_value(other.m_value)//shallow copy the pointer
        {
            //assert(m_value);
            //the pointer should never be invalid for the lifetime of the conditional
        }

        /**@brief returns the pointer*/
        short_t* value_ptr() const {
            assert(m_value);
            return m_value;
        }

        /**@brief returns the boolean condition*/
        short_t value() const {
            assert(m_value);
            return *m_value;
        }
    };

    template <typename T>
    struct is_conditional : boost::mpl::false_ {};

    template <uint_t Tag>
    struct is_conditional<conditional<Tag> >:boost::mpl::true_ {};

    template <uint_t Tag, uint_t SwitchId>
    struct is_conditional<switch_conditional<Tag, SwitchId> >:boost::mpl::true_ {};
}
