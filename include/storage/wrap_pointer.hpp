#pragma once
#include "../gridtools.hpp"

/**
@file
Write my documentation!
*/

/** This class wraps a raw pointer*/
namespace gridtools {

    template < typename T, bool Array = true >
    struct wrap_pointer {
	private:
		template <typename V>
		wrap_pointer(V);

	public:
        // TODO: turn into value_type?
        typedef T pointee_t;

		GT_FUNCTION
		wrap_pointer()
			: m_cpu_p(NULL), m_externally_managed(false) { }

        GT_FUNCTION
        wrap_pointer(wrap_pointer const &other)
            : m_cpu_p(other.m_cpu_p), m_externally_managed(true) { }

		GT_FUNCTION
		wrap_pointer(uint_t size, bool externally_managed) : m_externally_managed(externally_managed) {
			allocate_it(size);
#ifdef VERBOSE
			printf("CONSTRUCT pointer - %X %d\n", m_cpu_p, size);
#endif
		}

        GT_FUNCTION
        wrap_pointer(T* p, bool externally_managed)
            : m_cpu_p(p), m_externally_managed(externally_managed) { }

        GT_FUNCTION
        wrap_pointer<T>& operator=(T const& p) {
            m_cpu_p = &p;
            m_externally_managed = true;
            return *this;
        }

        GT_FUNCTION
        pointee_t *get() const { return m_cpu_p; }

        GT_FUNCTION
        T *get_pointer_to_use() { return m_cpu_p; }

        GT_FUNCTION
        void reset(T *cpu_p) { m_cpu_p = cpu_p; }

        GT_FUNCTION
        bool set_externally_managed(bool externally_managed_) { m_externally_managed = externally_managed_; }

        GT_FUNCTION
        bool is_externally_managed() const { return m_externally_managed; }

        GT_FUNCTION
        virtual ~wrap_pointer() {
#ifdef VERBOSE
#ifndef __CUDACC__
            std::cout << "deleting wrap pointer " << this << std::endl;
#endif
#endif
        }

        GT_FUNCTION
        void update_gpu() { assert(false); } //\todo find a way to remove this method

        GT_FUNCTION
        void update_cpu() { assert(false); } //\todo find a way to remove this method

        GT_FUNCTION
        void allocate_it(uint_t size) { 
			m_cpu_p = (Array) ? new T[size] : new T;
		}

        void free_it() {
            if (m_cpu_p && !m_externally_managed) {
#ifdef VERBOSE
#ifndef __CUDACC__
                std::cout << "deleting data pointer " << m_cpu_p << std::endl;
#endif
#endif
				// this conditional operator decides if it should call an 
				// array delete or a standard pointer delete operation.
				(Array) ? delete[] m_cpu_p : delete m_cpu_p;
                m_cpu_p = NULL;
            }
        }

        GT_FUNCTION
        operator T *() {
            assert(m_cpu_p);
            return m_cpu_p;
        }

        GT_FUNCTION
        operator T const *() const {
            assert(m_cpu_p);
            return m_cpu_p;
        }

        GT_FUNCTION
        T &operator[](uint_t i) {
            assert(m_cpu_p);
            return m_cpu_p[i];
        }

        GT_FUNCTION
        T const &operator[](uint_t i) const {
            assert(m_cpu_p);
            return m_cpu_p[i];
        }

        GT_FUNCTION
        T &operator*() {
            assert(m_cpu_p);
            return *m_cpu_p;
        }

        GT_FUNCTION
        T const &operator*() const {
            assert(m_cpu_p);
            return *m_cpu_p;
        }

        GT_FUNCTION
        T *operator+(uint_t i) {
            assert(m_cpu_p);
            return &m_cpu_p[i];
        }

        GT_FUNCTION
        T *const &operator+(uint_t i) const {
            assert(m_cpu_p);
            return &m_cpu_p[i];
        }

		GT_FUNCTION
		T *get_cpu_p() { return m_cpu_p; }

		GT_FUNCTION
		T *get_gpu_p() { assert(false); }

      protected:
        T *m_cpu_p;
        bool m_externally_managed;
    };

} // namespace gridtools
