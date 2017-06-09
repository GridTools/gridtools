## User Specified Extents

This is a feature to control compilation time in stencils with many stages and many data fields. This is a feature likely used by higher level approaches, since they may be already being doing data dependence analysis and extent computation, so they can specified those directly in the `make_computation` and save on C++ compilation time.

