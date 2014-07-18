
#ifndef USE_DOUBLE
#define USE_DOUBLE true
#else
#pragma message("USE_DOUBLE already defined, good!")
#endif


template <bool use_double, typename VT=double>
struct triple_t;

template <typename VT>
struct triple_t </*use_double=*/false, VT>{

  typedef triple_t<false, VT> data_type;

  VT _x,_y,_z;
  __host__ __device__ triple_t(VT a, VT b, VT c): _x(a), _y(b), _z(c) {}
  __host__ __device__ triple_t(): _x(-1), _y(-1), _z(-1) {}

  __host__ __device__ triple_t(triple_t<false,VT> const & t)
    : _x(t._x)
    , _y(t._y)
    , _z(t._z)
  {}

  triple_t<false,VT> floor() {
    VT m = std::min(_x, std::min(_y,_z));
    
    return (m==-1)?triple_t<false,VT>(m,m,m):*this;
  }

  VT x() const {return _x;}
  VT y() const {return _y;}
  VT z() const {return _z;}
};

template <typename VT>
struct triple_t</*use_double=*/true, VT> {

  typedef double data_type;

  double value;

  __host__ __device__ triple_t(int a, int b, int c): value(static_cast<long long int>(a)*100000000+static_cast<long long int>(b)*10000+static_cast<long long int>(c)) {}

  __host__ __device__ triple_t(): value(999999999999) {}

  __host__ __device__ triple_t(triple_t<true,VT> const & t)
    : value(t.value)
  {}

  triple_t<true, VT> floor() {
    if (x() == 9999 || y() == 9999 || z() == 9999) {
      return triple_t<true,VT>();
    } else {
      return *this;
    }
  }

  int x() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/100000000)%10000);
  }

  int y() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/10000)%10000);
  }

  template <typename T>
  int y(T& file) const {
    long long int cast = static_cast<long long int>(value);
    file << "$#$@%! " << cast << " " << static_cast<int>((cast/10000)%10000) << std::endl;
    return static_cast<int>((cast/10000)%10000);
  }

  int z() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast)%10000);
  }

};

template <bool V, typename T>
triple_t<V, T> operator*(int a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a*b.x(), a*b.y(), a*b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(int a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a+b.x(), a+b.y(), a+b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(triple_t<V, T> const& a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a.x()+b.x(), a.y()+b.y(), a.z()+b.z());
}

template <bool V, typename T>
std::ostream& operator<<(std::ostream &s, triple_t<V, T> const & t) { 
  return s << " (" 
           << t.x() << ", "
           << t.y() << ", "
           << t.z() << ") ";
}

template <bool V, typename T>
bool operator==(triple_t<V, T> const & a, triple_t<V, T> const & b) {
  return (a.x() == b.x() && 
          a.y() == b.y() &&
          a.z() == b.z());
}

template <bool V, typename T>
bool operator!=(triple_t<V, T> const & a, triple_t<V, T> const & b) {
  return !(a==b);
}

