#ifndef _DESCRIPTOR_BASE_H_
#define _DESCRIPTOR_BASE_H_

namespace gridtools {
    /**
       This defines the start_exchange, do_sends, etc, for all descriptors
    */
    template <typename HaloExch>
    struct descriptor_base {
        /**
           Type of the Level 3 pattern used. This is available only if the pattern uses a Level 3 pattern.
           In the case the implementation is not using L3, the type is not available.
        */
        typedef HaloExch pattern_type;
        typedef typename pattern_type::grid_type grid_type;

    private:
        grid_type const m_grid;
    public:
        pattern_type m_haloexch;//TODO private

        template <typename Array>
        descriptor_base(typename grid_type::period_type const &c, MPI_Comm const& comm, Array const* dimensions)
            :
            m_grid(grid_type(c,comm, dimensions)),
            m_haloexch(m_grid)
        {}

        descriptor_base(grid_type const &g)
            :
            m_grid(g),
            m_haloexch(g)
        {}

        /**
           function to trigger data exchange

           Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used, and vice versa.
        */
        void exchange() {
            m_haloexch.exchange();
        }

        /**
           function to trigger posting of receives when using split-phase communication.
        */
        void post_receives() {
            m_haloexch.post_receives();
        }

        /**
           function to perform sends (isend) of receives when using split-phase communication.
        */
        void do_sends() {
            m_haloexch.do_sends();
        }

        /**
           function to trigger data exchange initiation when using split-phase communication.

           Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used, and vice versa.
        */
        void start_exchange() {
            m_haloexch.start_exchange();
        }

        /**
           function to trigger data exchange

           Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used, and vice versa.
        */
        void wait() {
            m_haloexch.wait();
        }

        /**
           Retrieve the pattern from which the computing grid and other information
           can be retrieved. The function is available only if the underlying
           communication library is a Level 3 pattern. It would not make much
           sense otherwise.

           If used to get process grid information additional information can be
           found in \link GRIDS_INTERACTION \endlink
        */
        pattern_type const & pattern() const {return m_haloexch;}
        grid_type const & comm() const {return m_grid;}

    };
} // namespace gridtools



#endif
