#ifndef TIMERS_H
#define TIMERS_H

#include <mpi.h>
#include <cassert>

class Timers {
	public:
	double * total_time;
	private:
	double * start_time;
	public:

	typedef enum {
		TIMER_GLOBAL,
		TIMER_COMPUTE_STENCIL_INNER,
		TIMER_COMPUTE_STENCIL_BORDER,
		TIMER_COMPUTE_DOTPROD,
		TIMER_COMPUTE_RNG,
		TIMER_COMPUTE_MISC,
		TIMER_HALO_PACK,
		TIMER_HALO_ISEND_IRECV,
		TIMER_HALO_UNPACK_WAIT,
		TIMER_COMM_DOTPROD,
                TIMER_OBCS,                    // timer for the Open Bonudary Conditions (JURAJ)
		// Keep this entry at the end
		TIMER_LAST_ENTRY
	} timer_t;

	Timers() {
		total_time = new double[TIMER_LAST_ENTRY];
		start_time = new double[TIMER_LAST_ENTRY];
		for (int i=0; i<TIMER_LAST_ENTRY; i++) {
			total_time[i] = 0.0;
			start_time[i] = 0.0;
		}
	}

	~Timers() {
		delete [] total_time;
		delete [] start_time;
	}

	inline void start(int itimer) {
		assert(0 <= itimer && itimer < TIMER_LAST_ENTRY);
		start_time[itimer] = wall_time();
	}

	inline void stop(int itimer) {
		assert(0 <= itimer && itimer < TIMER_LAST_ENTRY);
		total_time[itimer] += wall_time() - start_time[itimer];
	}

	inline double wall_time() {
		return MPI_Wtime();
	}

	void print_timers() {
		printf("===========================================\n");
		printf("TIMER_GLOBAL                 = %.4e s\n", total_time[TIMER_GLOBAL]);
		printf("TIMER_COMPUTE_STENCIL_INNER  = %.4e s\n", total_time[TIMER_COMPUTE_STENCIL_INNER]);
		printf("TIMER_COMPUTE_STENCIL_BORDER = %.4e s\n", total_time[TIMER_COMPUTE_STENCIL_BORDER]);
		printf("TIMER_COMPUTE_DOTPROD        = %.4e s\n", total_time[TIMER_COMPUTE_DOTPROD]);
		printf("TIMER_COMPUTE_RNG            = %.4e s\n", total_time[TIMER_COMPUTE_RNG]);
		printf("TIMER_COMPUTE_MISC           = %.4e s\n", total_time[TIMER_COMPUTE_MISC]);
		printf("TIMER_HALO_PACK              = %.4e s\n", total_time[TIMER_HALO_PACK]);
		printf("TIMER_HALO_ISEND_IRECV       = %.4e s\n", total_time[TIMER_HALO_ISEND_IRECV]);
		printf("TIMER_HALO_UNPACK_WAIT       = %.4e s\n", total_time[TIMER_HALO_UNPACK_WAIT]);
		printf("TIMER_COMM_DOTPROD           = %.4e s\n", total_time[TIMER_COMM_DOTPROD]);
		printf("TIMER_OBCS                   = %.4e s\n", total_time[TIMER_OBCS]);
		printf("===========================================\n");
	}

};

#endif
