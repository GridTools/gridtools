boost::timer::cpu_times operator+(const boost::timer::cpu_times& t1, const boost::timer::cpu_times& t2 )
{
    boost::timer::cpu_times t;
    t.wall = t1.wall + t2.wall;
    t.user = t1.user + t2.user;
    t.system = t1.system+ t2.system;
    return t;
}