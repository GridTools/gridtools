========================
 User Specified Extents
========================

This is a feature to control compilation time in stencils with many
stages and many data fields. This is a feature likely used by higher
level approaches, since they may be already being doing data
dependence analysis and extent computation, so they can specified
those directly in the `make_computation` and save on C++ compilation
time.

A typical example of user code for making a `multi_stage` stencil is

.. code-block:: c++

 auto multi_stage = make_multistage(
    execute< forward >(),
    make_stage< lap_operator >(p_lap(), p_in()),
    make_stage< flx_operator >(p_flx(), p_in(), p_lap()),
    make_stage< fly_operator >(p_fly(), p_in(), p_lap()),
    make_stage< out_operator >(p_out(), p_in(), p_flx(), p_fly())
 );


The $\GT$ stencil composition library will take the multi_stage type
and compute (at compile time) at what extents each stencil operator
should be computed to make the computation correct. This computation
is non trivial and may stress the compiler substantially if the
computation is made of many stages and uses many data fields (this is
not the case of the example shown above, which is rather small).

The user can then decide to tell explicitly the library to use certain
extents. To do so the use can used `make_stage_with_extent` for
__**all**__ the stages in the computation. The corresponding
syntax for the previous example would be:

.. code-block:: c++

 auto multi_stage = make_multistage(
    execute< forward >(),
    make_stage_with_extent< lap_operator, extent<-1,1,-1,1> >(p_lap(), p_in()),
    make_stage_with_extent< flx_operator, extent<0,1,0,0> >(p_flx(), p_in(), p_lap()),
    make_stage_with_extent< fly_operator, extent<0,0,0,1> >(p_fly(), p_in(), p_lap()),
    make_stage_with_extent< out_operator, empty_extent>(p_out(), p_in(), p_flx(), p_fly())
 );


When the library finds that all the stages specify extents the extent
analysis is skipped. If only some stages specify extents a compilation
error is triggered explaining the situation.

The downside of using this approach is that is the extents used bigger
than what they should be, the computation would be inefficient. If the
specified extents are too small, a run-time error will be raised with
an access violation. Indeed, protecting the user at compile time for
this scenario would force the extent analysis to be performed and this
is exactly what the user wanted to avoid in the first place.

The prototype implementation does a check to see if all the stages in
the MSS components array uses extents, if so the computation of the
extents is avoided and the extents extracted form the ESF descriptors
and put in the extent map.

.. note::

 It would be very difficult to provide a default value for the
 extent (for instance for the final stage), since the placeholders are
 taken as variadic templates. It should be done by analysing the
 variadic pack but it seems to me an over-sophistication.

