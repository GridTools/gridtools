#pragma once
#include <boost/fusion/include/as_set.hpp>
//#include <boost/fusion/include/at_key.hpp>

namespace gridtools{

    template<typename Sequence>
    struct metadata_set{
        typedef typename boost::fusion::result_of::as_set<Sequence>::type set_t;

        DISALLOW_COPY_AND_ASSIGN(metadata_set);
        set_t m_set;

    public:
        metadata_set() : m_set(){};

        ~metadata_set(){
            boost::fusion::for_each(m_set, delete_it<set_t>(m_set));
        }

        template <typename T>
        void insert(T new_instance)
            {
                GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value), "error");
                assert(!present<T>());//must be uninitialized
                boost::fusion::at_key<T>(m_set)=new_instance;
            }

        /**
           @brief returns the raw pointer
        */
        template <typename T>
        T const& get()
            {
                GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value), "Internal error: calling metadata_set::get with a metadata type which has not been defined.");
                assert(present<T>());//must be initialized
                return boost::fusion::at_key<T>(m_set);
            }

        set_t& sequence_view() {return m_set;}

        template <typename T>
        bool present() {
            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value), "internal error: calling metadata_set::present with a metadata type which has not been defined");
            return boost::fusion::at_key<T>(m_set).get();
        }


        template<typename Set>
        struct delete_it{
        private:

            Set const& m_set;
        public:

            delete_it(Set const& set_) : m_set(set_){}

            template<typename T>
            void operator()(T t) const{
                //if(present<T>())
                    //delete boost::fusion::at_key<T>(m_set);
            }
        };


    };
}
