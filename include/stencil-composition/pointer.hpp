#pragma once

namespace gridtools{

    template<typename T>
    struct pointer{

    private:
        T const* m_t;
    public:
        typedef T value_type;
        pointer(): m_t(0){}
        pointer(T const* t_): m_t(t_){}

        void operator = (T const* t_){
            m_t=t_;
        }

        /**
           @brief returns the raw pointer
        */
        T const* get(){return m_t;}
        T const* operator -> () const {return m_t;}
        T operator * () const {return *m_t;}

    };
}
