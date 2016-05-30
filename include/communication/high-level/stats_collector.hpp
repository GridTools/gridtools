#ifndef _STATS_COLLECTOR_H_
#define _STATS_COLLECTOR_H_

#include <vector>
#include <map>
#include <set>
#include <numeric>
#include <iomanip>

#include <mpi.h>

#include "../common/array.hpp"
#include "../common/boollist.hpp"
#include "halo_descriptor.hpp"

namespace gridtools {

    // data structure for recording high-level communication events
    // high-level events correspond to api calls (pack, exchange, start_exchange, etc...)
    // high-level events are used to get an understanding of the overll communication pattern
    enum ExchangeEventType {
        ee_pack,
        ee_unpack,
        ee_exchange,
        ee_start_exchange,
        ee_wait,
        ee_post_receives,
        ee_do_sends
    };

    struct ExchangeEvent {
        ExchangeEvent(ExchangeEventType type, double start, double end, int fields, int pattern = -1)
            : type(type), wall_time_start(start), wall_time_end(end), fields(fields), pattern(pattern){};

        ExchangeEventType type;
        double wall_time_start;
        double wall_time_end;
        int fields;
        int pattern;
    };

    // data structure for recording low-level communication events
    // low-level events correspond to MPI calls (MPI_wait, MPI_Isend, etc...)
    // low-level events can be used to trace all MPI calls
    enum CommEventType { ce_send, ce_receive, ce_send_wait, ce_receive_wait };

    struct CommEvent {
        CommEvent(CommEventType t, int other, int tg, int size, double start, double end, int pat = -1)
            : type(t), other_rank(other), tag(tg), message_size(size), wall_time_start(start), wall_time_end(end),
              pattern(pat){};

        CommEventType type;
        int other_rank;
        int tag;
        int message_size;
        double wall_time_start;
        double wall_time_end;
        int pattern;
    };

    // data structure used to store enough information about a communication pattern
    // for it to be replicated
    enum PatternType { pt_dynamic, pt_generic }; // note that only pt_dynamic is supported for now

    template < int DIM >
    struct Pattern {
        typedef array< halo_descriptor, DIM > halo_array;
        typedef gcl_utils::boollist< DIM > ptype;

        std::vector< int > proc_map;
        PatternType type;
        halo_array halos;
        bool periodicity[DIM];
        int coords[DIM];
        int dims[DIM];

        Pattern(PatternType t,
            const halo_array &h,
            std::vector< int > map,
            const ptype &c,
            int coords_[DIM],
            int dims_[DIM])
            : type(t), halos(h), proc_map(map) {
            c.copy_out(periodicity);
            std::copy(coords_, coords_ + DIM, coords);
            std::copy(dims_, dims_ + DIM, dims);
        };

        const halo_descriptor &halo(int h) const {
            assert(h < DIM);
            return halos[proc_map[h]];
        }
    };

    // singleton for collecting run time statistics about communication
    template < int DIM >
    class stats_collector {
      public:
        typedef stats_collector< DIM > collector;
        typedef std::vector< CommEvent >::iterator event_iterator;
        typedef std::vector< CommEvent >::const_iterator const_event_iterator;
        typedef std::vector< ExchangeEvent >::iterator exchange_iterator;
        typedef std::vector< ExchangeEvent >::const_iterator const_exchange_iterator;
        typedef typename std::vector< Pattern< DIM > >::iterator pattern_iterator;
        typedef typename std::vector< Pattern< DIM > >::const_iterator const_pattern_iterator;

        // get instance of the stats_collector singleton
        static collector *instance() { return instance_ ? instance_ : (instance_ = new collector); }

        // intialized the singleton
        // performs a syncronization across all MPI processes and records a time stamp
        // with which to normalize all subsequent time stamps across the MPI processes
        void init(MPI_Comm comm) {
            if (initialized_)
                return;

            comm_ = comm;
            // perform barrier syncronization
            MPI_Barrier(comm_);
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);

            // get initial time stamp
            initial_time_stamp_ = MPI_Wtime();

            initialized_ = true;
        }

        // add a low-level MPI event
        void add_event(const CommEvent &event) {
            if (recording_)
                events_.push_back(event);
        }

        // add a high-level exchange event
        void add_event(const ExchangeEvent &event) {
            if (recording_)
                exchange_events_.push_back(event);
        }

        int add_pattern(const Pattern< DIM > &pat) {
            patterns_.push_back(pat);
            return patterns_.size() - 1;
        }

        int num_patterns() const { return patterns_.size(); }

        // return number of events recorded thus far
        int num_events() const { return events_.size(); }

        // functions providing read-only access to the high level events
        const_exchange_iterator exchange(int index) const {
            assert(index < exchange_events_.size());
            return exchange_events_.begin() + index;
        }
        const_exchange_iterator exchange_begin() const { return exchange_events_.begin(); }
        const_exchange_iterator exchange_end() const { return exchange_events_.end(); }

        // functions providing read-only access to the events
        const_event_iterator event(int index) const {
            assert(index < events_.size());
            return events_.begin() + index;
        }
        const_event_iterator events_begin() const { return events_.begin(); }
        const_event_iterator events_end() const { return events_.end(); }

        // functions providing read-only access to the patterns
        const_pattern_iterator pattern(int index) const {
            assert(index < patterns_.size());
            return patterns_.begin() + index;
        }
        const_pattern_iterator patterns_begin() const { return patterns_.begin(); }
        const_pattern_iterator patterns_end() const { return patterns_.end(); }

        // toggle recording on or off. recording is set to false, calls to add_event() are ignored.
        void recording(bool state) { recording_ = state; }

        // print information about communicatio pattern that is required
        // to reproduce communication
        template < typename S >
        void evaluate(S &stream, int level = 0) const {
            // determine which subset of patterns were actually used
            std::set< int > patterns_used;
            for (const_exchange_iterator it = exchange_begin(); it != exchange_end(); it++)
                patterns_used.insert(it->pattern);

            // enumerate the patterns from 0:patterns_used.size()-1
            std::map< int, int > pattern_map;
            for (std::set< int >::const_iterator it = patterns_used.begin(); it != patterns_used.end(); it++)
                pattern_map[*it] = pattern_map.size() - 1;

            // stream << "global inded, local index, minus, plus, begin, end, total_length" << std::endl;
            if (!rank)
                stream << "patterns_start" << std::endl;
            for (std::map< int, int >::const_iterator it = pattern_map.begin(); it != pattern_map.end(); it++) {
                int p = it->first;
                if (!rank) {
                    stream << "pattern " << it->second << "  " << it->first << std::endl;
                    stream << "       IJK ";
                    for (int i = 0; i < DIM; i++)
                        stream << " " << patterns_[p].proc_map[i];
                    stream << std::endl << "  periodic ";
                    for (int i = 0; i < DIM; i++)
                        stream << " " << (patterns_[p].periodicity[i] ? "T" : "F");
                    stream << std::endl << "      grid ";
                    for (int i = 0; i < DIM; i++)
                        stream << " " << patterns_[p].dims[i];
                    stream << std::endl;
                }
                for (int i = 0; i < DIM; i++) {
                    // allocate memory for receive buffer
                    std::vector< int > coord_buff(size);
                    int dim = patterns_[p].coords[i];
                    MPI_Gather(&dim, 1, MPI_INT, &coord_buff[0], 1, MPI_INT, 0, comm_);

                    if (!rank) {
                        stream << " coords(" << i << ") ";
                        for (int dim = 0; dim < size; dim++)
                            stream << " " << coord_buff[dim];
                        stream << std::endl;
                    }
                }
                for (int i = 0; i < DIM; i++) {
                    // allocate memory for receive buffer
                    std::vector< int > recvbuff(size);

                    // gather and output the minus ranges
                    int minus = patterns_[p].halo(i).minus();
                    MPI_Gather(&minus, 1, MPI_INT, &recvbuff[0], 1, MPI_INT, 0, comm_);
                    if (!rank) {
                        stream << "\tminus\t";
                        for (int r = 0; r < size; r++)
                            stream << recvbuff[r] << " ";
                        stream << std::endl;
                    }
                    // gather and output the plus ranges
                    int plus = patterns_[p].halo(i).plus();
                    MPI_Gather(&plus, 1, MPI_INT, &recvbuff[0], 1, MPI_INT, 0, comm_);
                    if (!rank) {
                        stream << "\tplus \t";
                        for (int r = 0; r < size; r++)
                            stream << recvbuff[r] << " ";
                        stream << std::endl;
                    }
                    // gather and output the begin ranges
                    int begin = patterns_[p].halo(i).begin();
                    MPI_Gather(&begin, 1, MPI_INT, &recvbuff[0], 1, MPI_INT, 0, comm_);
                    if (!rank) {
                        stream << "\tbegin\t";
                        for (int r = 0; r < size; r++)
                            stream << recvbuff[r] << " ";
                        stream << std::endl;
                    }
                    // gather and output the end ranges
                    int end = patterns_[p].halo(i).end();
                    MPI_Gather(&end, 1, MPI_INT, &recvbuff[0], 1, MPI_INT, 0, comm_);
                    if (!rank) {
                        stream << "\tend  \t";
                        for (int r = 0; r < size; r++)
                            stream << recvbuff[r] << " ";
                        stream << std::endl;
                    }
                    // gather and output the total_length ranges
                    int total_length = patterns_[p].halo(i).total_length();
                    MPI_Gather(&total_length, 1, MPI_INT, recvbuff.data(), 1, MPI_INT, 0, comm_);
                    if (!rank) {
                        stream << "\ttotal\t";
                        for (int r = 0; r < size; r++)
                            stream << recvbuff[r] << " ";
                        stream << std::endl;
                    }
                }
            }
            if (!rank) {
                stream << "patterns_end" << std::endl << "exchanges_start" << std::endl;
                for (const_exchange_iterator it = exchange_begin(); it != exchange_end(); it++) {
                    if (it->type == ee_pack)
                        stream << pattern_map[it->pattern] << "\tstart\t" << it->fields << "\tfields" << std::endl;
                    else if (it->type == ee_unpack)
                        stream << pattern_map[it->pattern] << "\tstop\t" << it->fields << "\tfields" << std::endl;
                }
                stream << "exchanges_end" << std::endl;
            }

            if (!rank)
                stream << "pattern_times_start" << std::endl;

            typedef std::map< ExchangeEventType, double > PatternTimeTable;
            PatternTimeTable time_table;
            time_table[ee_pack] = time_table[ee_unpack] = time_table[ee_wait] = time_table[ee_exchange] =
                time_table[ee_start_exchange] = 0.;
            // a map that stores a time table for each pattern
            std::map< int, PatternTimeTable > pattern_times;
            // initialize time table for each pattern to zero
            for (std::map< int, int >::const_iterator it = pattern_map.begin(); it != pattern_map.end(); it++) {
                pattern_times[it->first] = time_table;
            }
            for (std::vector< ExchangeEvent >::const_iterator it = exchange_events_.begin();
                 it != exchange_events_.end();
                 it++) {
                double dt = it->wall_time_end - it->wall_time_start;
                pattern_times[it->pattern][it->type] += dt;
            }

            // storage for MPI reductions
            double local_times[5];
            double mean_times[5];
            // temporary storage for output string
            std::vector< char > str_storage(256);
            char *str = &str_storage[0];

            // print details of the communication patterns
            if (!rank) {
                stream << ".-----------------------------------------------------------------------------------."
                       << std::endl;
                sprintf(
                    str, "|%7s |%14s%14s%14s%14s |%14s  |", "pattern", "pack", "wait", "start_exchg", "exchg", "TOTAL");
                stream << str << std::endl;
                stream << ".-----------------------------------------------------------------------------------."
                       << std::endl;
            }
            double sum_pack = 0., sum_wait = 0., sum_exchg = 0., sum_start_exchg = 0.;
            int idx = 0;
            for (std::map< int, PatternTimeTable >::iterator it = pattern_times.begin(); it != pattern_times.end();
                 it++) {
                // local_times[0,1,2,3] = pack, wait, start_exchange, exchange times
                // local_times[4] = sum of all times
                local_times[0] = it->second[ee_pack] + it->second[ee_unpack];
                local_times[1] = it->second[ee_wait];
                local_times[2] = it->second[ee_start_exchange];
                local_times[3] = it->second[ee_exchange];
                local_times[4] = std::accumulate(local_times, local_times + 4, 0.);

                sum_pack += local_times[0];
                sum_wait += local_times[1];
                sum_start_exchg += local_times[2];
                sum_exchg += local_times[3];

                MPI_Reduce(local_times, mean_times, 5, MPI_DOUBLE, MPI_SUM, 0, comm_);

                if (!rank) {
                    sprintf(str,
                        "|%7d |%14.8f%14.8f%14.8f%14.8f | %14.8f |",
                        idx++,
                        mean_times[0] / double(size),
                        mean_times[1] / double(size),
                        mean_times[2] / double(size),
                        mean_times[3] / double(size),
                        mean_times[4] / double(size));
                    stream << str << std::endl;
                }
            }
            if (!rank)
                stream << ".-----------------------------------------------------------------------------------."
                       << std::endl;
            // output the cumulative communication times across all patterns
            // requires MPI reduction
            local_times[0] = sum_pack;
            local_times[1] = sum_wait;
            local_times[2] = sum_start_exchg;
            local_times[3] = sum_exchg;
            local_times[4] = std::accumulate(local_times, local_times + 4, 0.);
            MPI_Reduce(local_times, mean_times, 5, MPI_DOUBLE, MPI_SUM, 0, comm_);
            if (!rank) {
                sprintf(str,
                    "%8s |%14.8f%14.8f%14.8f%14.8f | %14.8f |",
                    "SUM_MEAN",
                    mean_times[0] / double(size),
                    mean_times[1] / double(size),
                    mean_times[2] / double(size),
                    mean_times[3] / double(size),
                    mean_times[4] / double(size));
                stream << str << std::endl;
            }
            MPI_Reduce(local_times, mean_times, 5, MPI_DOUBLE, MPI_MIN, 0, comm_);
            if (!rank) {
                sprintf(str,
                    "%7s  |%14.8f%14.8f%14.8f%14.8f | %14.8f |",
                    "SUM_MIN",
                    mean_times[0],
                    mean_times[1],
                    mean_times[2],
                    mean_times[3],
                    mean_times[4]);
                stream << str << std::endl;
            }
            MPI_Reduce(local_times, mean_times, 5, MPI_DOUBLE, MPI_MAX, 0, comm_);
            if (!rank) {
                sprintf(str,
                    "%7s  |%14.8f%14.8f%14.8f%14.8f | %14.8f |",
                    "SUM_MAX",
                    mean_times[0],
                    mean_times[1],
                    mean_times[2],
                    mean_times[3],
                    mean_times[4]);
                stream << str << std::endl;
                stream << "         .--------------------------------------------------------------------------."
                       << std::endl;
            }

            if (!rank)
                stream << "pattern_times_end" << std::endl;
        }

        // print the profiling results to stdout
        // the parameter level specifies how detailed the printout should be
        //      0 : show only the communication patterns
        //      1 : show patterns and exchange events
        //      2 : show all profiling information: patters, exchange events,
        //          and low-level MPI calls
        template < typename S >
        void print(S &stream, int level = 1) const {
            // map of event types onto their names for printing
            std::map< CommEventType, std::string > event_labels;
            event_labels[ce_send] = std::string("send");
            event_labels[ce_receive_wait] = std::string("receive_wait");
            event_labels[ce_send_wait] = std::string("send_wait");
            event_labels[ce_receive] = std::string("receive");

            std::map< ExchangeEventType, std::string > exchange_labels;
            exchange_labels[ee_pack] = std::string("pack");
            exchange_labels[ee_unpack] = std::string("unpack");
            exchange_labels[ee_exchange] = std::string("exchange");
            exchange_labels[ee_start_exchange] = std::string("start exchg");
            exchange_labels[ee_wait] = std::string("wait");

            std::map< PatternType, std::string > pattern_labels;
            pattern_labels[pt_dynamic] = std::string("dynamic");
            pattern_labels[pt_generic] = std::string("generic");

            // temporary storage for output string
            std::vector< char > str_storage(256);
            char *str = &str_storage[0];

            // print details of the communication patterns
            stream << "===============================================================================" << std::endl;
            stream << " PATTERNS " << std::endl;
            sprintf(str, "%8s%8s%3s%3s%3s", "pattern", "type", "I", "J", "K");
            stream << str << std::endl;
            stream << "-------------------------------------------------------------------------------" << std::endl;
            int idx = 0;
            for (const_pattern_iterator it = patterns_begin(); it != patterns_end(); it++, idx++) {
                stream << idx << "\t " << pattern_labels[it->type];
                for (int i = 0; i < DIM; i++)
                    stream << "  " << it->proc_map[i];
                stream << std::endl;
                for (int i = 0; i < DIM; i++) {
                    const halo_descriptor &h = it->halo(i);
                    stream << "\t\t" << h << std::endl;
                }
            }
            if (level == 2) {
                // print low-level MPI communication details
                stream << "==============================================================================="
                       << std::endl;
                stream << " COMMS " << std::endl;
                sprintf(str,
                    "%8s%6s%14s%5s%6s%12s%11s%13s",
                    "pattern",
                    "idx",
                    "type",
                    "rank",
                    "tag",
                    "size",
                    "t_start",
                    "t_duration");
                stream << str << std::endl;
                stream << "-------------------------------------------------------------------------------"
                       << std::endl;

                // print each event in order of collection
                idx = 0;
                for (const_event_iterator it = events_begin(); it != events_end(); it++, idx++) {
                    sprintf(str,
                        "%8d%6d%14s%5d%6d%12d%11.5f%13.10f",
                        it->pattern,
                        idx,
                        event_labels[it->type].c_str(),
                        it->other_rank,
                        it->tag,
                        it->message_size,
                        it->wall_time_start - initial_time_stamp_,
                        it->wall_time_end - it->wall_time_start);
                    stream << str << std::endl;
                }
            }
            if (level >= 1) {
                stream << "==============================================================================="
                       << std::endl;
                stream << " EXCHANGE EVENTS " << std::endl;
                sprintf(str, "%8s%6s%14s%7s%11s%11s", "pattern", "idx", "type", "fields", "t_start", "t_duration");
                stream << str << std::endl;
                stream << "-------------------------------------------------------------------------------"
                       << std::endl;
                idx = 0;
                for (const_exchange_iterator it = exchange_begin(); it != exchange_end(); it++, idx++) {
                    sprintf(str,
                        "%8d%6d%14s%7d%11.5f%11.8f",
                        it->pattern,
                        idx,
                        exchange_labels[it->type].c_str(),
                        it->fields,
                        it->wall_time_start - initial_time_stamp_,
                        it->wall_time_end - it->wall_time_start);
                    stream << str << std::endl;
                }
            }
            stream << "===============================================================================" << std::endl;
        }

      private:
        stats_collector() : recording_(false), initialized_(false) {
            // reserve space for storing events and patterns to avoid memory
            // allocation overheads during profiling
            events_.reserve(1023);
            exchange_events_.reserve(1023);
            patterns_.reserve(63);
        };
        stats_collector(collector const &){};

        // instance of singleton
        static collector *instance_;

        // time stamp after MPI syncronization at initialization
        // all subsequently stored time values are relative to this
        double initial_time_stamp_;

        // list of all recorded events
        std::vector< CommEvent > events_;
        std::vector< ExchangeEvent > exchange_events_;

        // flag whether to record events
        bool recording_;

        // flag whether initialized
        bool initialized_;

        std::vector< Pattern< DIM > > patterns_;

        // storage for MPI information
        int rank;
        int size;
        MPI_Comm comm_;
    };

    // make conventient handles for collectors to avoid cumbersome instance()-> interface
    extern stats_collector< 2 > &stats_collector_2D;
    extern stats_collector< 3 > &stats_collector_3D;

} // namespace gridtools

#endif
