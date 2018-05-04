/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

/** \defgroup GCL Generic Communication Layer
 */

/** \addtogroup GCL
 * @{
 */

/**
Generic Communicatin Library Reference Manual
<UL>
<LI> \ref gcl_intro
     <BR>&nbsp;&nbsp;&nbsp;<em>Introduction</em>
<LI> \ref GCL_L3
     <BR>&nbsp;&nbsp;&nbsp;<em>GCL Level 3</em>
<LI> \ref GCL_L2
     <BR>&nbsp;&nbsp;&nbsp;<em>GCL Level 2</em>
<LI> \ref GCL_L1
     <BR>&nbsp;&nbsp;&nbsp;<em>GCL Level 1</em>
</UL>
*/

/** \page gcl_intro Introduction
<span
style='background:white'>The Generic Communication Library (GCL) is organized
in three levels that will be described below here. The general idea is,
however, to produce a <b>library of communication patterns</b> each of which
providing the three layers. Each pattern is associated with a <i>topology class</i>
which is used to put processes/threads in relation with each other. It is not
necessary that different patterns share common concepts and interfaces, even
though uniformity should be a target. Given a pattern, up to three versions of
it may exist, each corresponding to a level of the abstraction hierarchy. They
can be used independently of each other. </span></p>
<p class=Standard><span style='background:white'>The main idea is that level 3
is the lowest level which maps to current communication mechanisms, like MPI.
It can be roughly identified with the <i>transport layer</i> in a communication
network. It assumes data is transferred though a <b>high-latency medium</b> so <b>buffering
is essential</b>. Actually what level 3 does is to <b>take already filled
buffers and exchange them</b> according to the communication pattern. </span></p>
<p class=Standard><span style='background:white'>Level 3 can be used in other
scenarios, but it can result in poor performance since unnecessary copies may
be required, since the <b>user must buffer data explicitly</b>. </span></p>
<p class=Standard><span style='background:white'>&nbsp;</span></p>
<p class=Standard><b><span style='background:white'>Level 2 assumes certain
data layouts and distributions</span></b><span style='background:white'>, and
perform the patterns according to this knowledge. As an example, consider the
halo exchange in a stencil application for regular grids. The big problem grids
are partitioned into smaller grids, each of those processed by a process, a
thread, or something else. This is the domain decomposition scenario. The
decomposition is assumed to be uniform, so each process/thread has the same
abstraction over the data. Level 2 knows about it and <b>takes the description
of the data to be exchanged and performs the data exchange</b>. It may or no
use the level 3, since buffering may be not required. Several communication
mechanisms allow for semi-shared memory view of the address space. This may
require the grids <b>memory be registered with communication pattern</b>, so
that a mapping of the addresses may allow remote writes of elements. Since this
is generic, the registration should happen anyway, since the api should work on
every <i>transport layer</i>. This is indication of the problem: <b>since
communication mechanisms and tools are so diverse, the generic API can end up
in requiring lot of information to deal with all possibilities.</b>
Alternatively, <b>the choice may be to provide initialization procedures that
are architecture dependent</b>. The initialization problem is probably the more
complex to approach in a generic way, and I think we need to be very pragmatic.</span></p>
<p class=Standard><span style='background:white'>&nbsp;</span></p>
<p class=Standard><b><span style='background:white'>Level 1</span></b><span
style='background:white'> is similar to level2, only that the <b>user must
provide</b>, instead of description of memory areas, <b>functions to extract
and insert data in those areas</b>. Actually, level 2 can be implemented with
level 1 (Now, the story of the levels comes from early discussions, and should
change into something more reasonable). In figure a showed an
container+iterator interface, which seems to be the most promising. The
iterators must contain all the logic to pick the right values, thus jumping
around memory as needed. This approach may be used to produce communication
patters for more general applications than stencil application on regular
grids. <b>Level 1, as level 2, can take advantage of level 3 when used on top
of a high latency interconnect</b>.</span></p>
*/

/** \page GCL_L3 GCL Level 3
<h4 class="c4">Processor Grid</h4><p class="c4"><span class="c5">Any two dimensional processor grid object must support
the following member functions:</span></p><p class="c1"><span class="c5"></span></p><table border="1" cellpadding="0"
cellspacing="0" class="c20"><tr><td class="c12"><p class="c11 c4"><span class="c0">void dims(int &amp;I, int &amp;J)
const</span></p></td><td class="c2"><p class="c1 c11"><span class="c5"></span></p></td><td class="c6"><p class="c11
c4"><span class="c5">Retreive grid dimensions</span></p></td></tr><tr><td class="c12"><p class="c11 c4"><span
class="c0">void coords(int &amp;i, int &amp;j) const</span></p></td><td class="c2"><p class="c1 c11"><span
class="c5"></span></p></td><td class="c6"><p class="c11 c4"><span class="c5">Retreive calling process
coordinates</span></p></td></tr><tr><td class="c12"><p class="c11 c4"><span class="c0">template &lt;int II, int JJ&gt;
int proc() const</span></p></td><td class="c2"><p class="c1 c11"><span class="c5"></span></p></td><td class="c6"><p
class="c11 c4"><span class="c5">Process ID of process in (i+II, j+JJ) &nbsp; &nbsp; &nbsp;
(*)</span></p></td></tr><tr><td class="c12"><p class="c11 c4"><span class="c0">int proc(int ii, int jj)
const</span></p></td><td class="c2"><p class="c1 c11"><span class="c5"></span></p></td><td class="c6"><p class="c11
c4"><span class="c5">Process ID of process in (i+ii, j+jj) &nbsp; &nbsp; &nbsp; (*)</span></p></td></tr>
</table><p class="c4"><span class="c5">(*) (i,j) are assumed to be the coordinates of the calling process.</span></p><p
class="c1"><span class="c5"></span></p><p class="c4"><span class="c5">It is assumed the first dimension is i and the
second is j. Construction of the process grid is dependent on the actual class used. The list of the processor grisd
available in GCL will be provided with the manual.</span></p><p class="c1"><span class="c5"></span></p><p
class="c4"><span class="c5">Any three dimensional processor grid object must support the following member
functions:</span></p><p class="c1"><span class="c5"></span></p><table border="1" cellpadding="0" cellspacing="0"
class="c20"><tr class="c7"><td class="c12"><p class="c11 c4"><span class="c0">void dims(int &amp;I, int &amp;J, int
&amp;K) const</span></p></td><td class="c2"><p class="c1 c11"><span class="c5"></span></p></td><td class="c6"><p
class="c11 c4"><span class="c5">Retreive grid dimensions</span></p></td></tr><tr class="c7"><td class="c12"><p
class="c11 c4"><span class="c0">void coords(int &amp;i, int &amp;j, int &amp;k) const</span></p></td><td class="c2"><p
class="c1 c11"><span class="c0"></span></p></td><td class="c6"><p class="c4 c11"><span class="c0">Retreive calling
process coordinates</span></p></td></tr><tr class="c7"><td class="c12"><p class="c11 c4"><span class="c0">template
&lt;int II, int JJ, int KK&gt; int proc() const</span></p></td><td class="c2"><p class="c1 c11"><span
class="c5"></span></p></td><td class="c6"><p class="c11 c4"><span class="c5">Process ID of process in (i+II, j+JJ,k+KK)
&nbsp; &nbsp; &nbsp; (*)</span></p></td></tr><tr class="c7"><td class="c12"><p class="c11 c4"><span class="c0">int
proc(int ii, int jj, int kk) const</span></p></td><td class="c2"><p class="c1 c11"><span class="c5"></span></p></td><td
class="c6"><p class="c11 c4"><span class="c5">Process ID of process in (i+ii, j+jj, j+kk) &nbsp; &nbsp; &nbsp;
(*)</span></p></td></tr></table><p class="c4"><span class="c5">(*) (i,j,k) are assumed to be the coordinates of the
calling process.</span></p><p class="c1"><span class="c5"></span></p><h4 class="c4"><a
name="h.k2kd75ufcyzl"></a>Processor grids in GCL</h4><p class="c4"><span class="c5">There are four processor grids types
immediately available in GCL. The first two are </span><span class="c0">_2D_process_grid_t</span><span
class="c5">&nbsp;and </span><span class="c0">_3D_process_grid_t</span><span class="c5">, which provide the interface to
create the grid (no construcors)</span></p><p class="c1"><span class="c5"></span></p><p class="c4"><span class="c0">void
create(int P, int pid)</span></p><p class="c1"><span class="c5"></span></p><p class="c4"><span class="c5">where P is the
number of processes available and pid is the pid of the calling process.</span></p><p class="c1"><span
class="c5"></span></p><p class="c4"><span class="c5">The other two are made to make use of a MPI_CART already created.
The classes are </span></p><p class="c4"><span class="c0">MPI_2D_process_grid_t</span><span class="c5">&nbsp;and
</span><span class="c0">MPI_3D_process_grid_t</span><span class="c5">&nbsp;with the following interfaces</span></p><p
class="c1"><span class="c5"></span></p><p class="c4"><span class="c0">MPI_2D_process_grid_t(MPI_Comm comm)</span><span
class="c5">, a constructor that receives the communicoator associated with the CART</span></p><p class="c1"><span
class="c5"></span></p><p class="c4"><span class="c0">void create(MPI_Comm comm)</span><span class="c5">&nbsp; a creation
function hat receives the communicoator associated with the CART</span></p><p class="c1"><span class="c5"></span></p>
<h4 class="c0"><a name="h.cjmgdgdnspyc"></a>Communication Pattern</h4><p class="c0"><span class="c8">Communication
pattern has one template argument that is the processor grid, and interfaces to associate the buffers to the
communication </span><span class="c8 c13">channels</span><span class="c8">.</span></p><p class="c0 c10"><span
class="c8"></span></p><p class="c0"><span class="c8 c12">Halo_Exchange_2D&lt;PROC_GRID&gt;(proc_grid) he;</span></p><p
class="c0"><span class="c8 c12">he.register_buffers(send_buffers, receive_buffers);</span></p><p class="c0"><span
class="c8 c12">he.execute();</span></p><p class="c0 c10"><span class="c8"></span></p><p class="c0"><span class="c8">Send
and receive buffers are classes that contain the buffers pointers and sizes. Alternatively, buffers can be registered
one by one as follows:</span></p><p class="c0 c10"><span class="c8"></span></p><p class="c0"><span class="c8
c12">he.register_send_to_buffer&lt;I,J&gt;(pointer_to_buffer, size);</span></p><p class="c0"><span class="c8
c12">he.register_receive_from_buffer&lt;I,J&gt;(pointer_to_buffer, size);</span></p><p class="c0 c10"><span
class="c8"></span></p><p class="c0"><span class="c8">where I and J are the relative coordinates of the processors that
are destination or sources of the data, respectively.</span></p><p class="c0 c10"><span></span></p><p class="c0
c10"><span></span></p><table border="1" cellpadding="0" cellspacing="0" class="c5"><tr><td class="c7"><p class="c6
c0"><span class="c8 c12">Halo_Exchange_2D&lt;PROC_GRID&gt;(proc_grid)</span></p></td><td class="c3"><p class="c6
c0"><span>Contructor. Proc_grid must be a processor grid as decsribed above</span></p></td></tr><tr><td class="c7"><p
class="c6 c0"><span class="c8 c12">template &lt;int I, int J&gt; void register_send_to_buffer&lt;I,J&gt;(void* ptr, int
size)</span></p></td><td class="c3"><p class="c6 c0"><span>Register with the pattern the bufferfor sending data to
process (i+I, j+J) &nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span class="c8 c12">void
register_receive_from_buffer&lt;I,J&gt;(void* ptr, int size, int I, int J)</span></p></td><td class="c3"><p class="c6
c0"><span>Register with the pattern the buffer for receiving data from process (i+I, j+J)
&nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span class="c8 c12">template &lt;int I, int J&gt;
void set_send_to_size&lt;I,J&gt;(int size)</span></p></td><td class="c3"><p class="c6 c0"><span>Set buffer size for
buffert to be sent to process (i+I, j+J) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span
class="c8 c12">void set_send_to_size(int size, int I, int J)</span></p></td><td class="c3"><p class="c6 c0"><span>Set
buffer size for buffert to be sent to process (i+I, j+J) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6
c0"><span class="c8 c12">template &lt;int I, int J&gt; void set_receive_from_size&lt;I,J&gt;(int
size)</span></p></td><td class="c3"><p class="c6 c0"><span>Set buffer size for buffert to be received from &nbsp;process
(i+I, j+J) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span class="c8 c12">void set_receive_from_size(int
size, int I, int J)</span></p></td><td class="c3"><p class="c6 c0"><span>Set buffer size for buffert to be received from
&nbsp;process (i+I, j+J) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp;(*)</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span>execute() const</span></p></td><td class="c3"><p
class="c6 c0"><span>Function to trigger data exchange.</span></p></td></tr><tr><td class="c7"><p class="c6
c0"><span>pre_execute() const</span></p></td><td class="c3"><p class="c6
c0"><span>NECESSARY?</span></p></td></tr><tr><td class="c7"><p class="c6 c0"><span>post_execute()
const</span></p></td><td class="c3"><p class="c6 c0"><span>NECESSARY?</span></p></td></tr></table><p class="c0"><span
class="c8">(*) (i,j) are assumed to be the coordinates of the calling process.</span></p><p class="c0
c10"><span></span></p><p class="c0"><span class="c17 c22">IMPORTANT</span><span
class="c17">:</span><span>&nbsp;</span><span class="c17">The pattern is assumed to be </span><span class="c13
c17">regular</span><span>. This means that all processes/threads are assumed to communicate with the same neighbors and
with the same sizes. In a non-cyclic case the proper adjustment has to be made by the processes at the boundaries. This
assumption is necessary to perform data exchange without exchanging empty messages. Other cases can be handled by other
communication patterns, for instance an irregular_halo_exchange pattern in which each process can communicate with
arbitrary neighbors with arbitrary sizes.</span></p>
*/

/** \page GCL_L2 GCL Level 2
    Level 2 introduces some application dependent concepts, such as
    data layout, multidimensional arrays, domain decomposition,
    etc. For this reason, Level 2 less less a communication library
    and more a application development library.
    There are two patterns available at Level 2 at the moment. One is
    a halo exchange for regular multidimensiona grids (2 and 3D
    presenlty) and the other is a generic all-to-all.
    \section L2-HALO_EX Halo Exchange for regular grids
    The Halo exchange pattern is implemented by \link
    gridtools::halo_exchange_dynamic_ut \endlink class. This class takes 4
    template arguments, the first two of which are worthy being
    explored here in detail.
    Before we proceed, we make clear that this is the main class for
    the halo exchange pattern in the case in which the data pointers
    are not known before hand and they are passed to the pattern when
    packing and unpacking is needed, and it is the only pattern
    provided at this level for the moment, since validation and
    feedback from the users should be evaluated before implementing
    other similar solutions.
    The interface requires two layout maps template arguments ( \link
    gridtools::layout_map \endlink ) one to specify the data layout, the
    other to specify the relation between data layout and processor
    grid. This is an important asepct that will be explained here and
    also in the introduction.
    The First layout map to be passed to the pattern class is the
    data layout. The user defines a convention in which the
    dimensions of the data fields are ordered logically depending on
    the application and/or user preferences. For instance, we can
    call the dimensions in this application order i, j, and k. The
    the layout map in this case specifies in what position each
    dimension is in the increasing stride order. For instance:
    \code
    gridtools::layout_map<1,0,2>
    \endcode
    Indicates that the first dimension in the data (i) is the second
    in the increasing stride order, while the second (j) is actually
    the first one (has stride 1). Finally the k dimension is the one
    with the highest stride and is in the third position (2). The
    layout in memory of the user data can be specified then as 'jik'.
    Similarly, the second template argument in the halo exchange
    pattern is the map between data coordinates and the processor
    grid coordinates. The following layout specification
    \code
    gridtools::layout_map<1,0,2>
    \endcode
    would mean: The first dimension in data matches with the second
    dimension of the computing grid, the second is the first, and the
    third one is the third one.
    Let's consider a 2D case at first, to show an additional
    example. Suppose user data is thought to be ij, meaning the user
    think to i as the first coordinate and j to the
    second. Alternatively the user would use (i,j) to indicate an
    element in the data array. The layout is C like, so that j is
    actuallly the first coordinate in the increasing stride ordering,
    and i the second.
    The first template argument to the pattern would then be
    \code
    gridtools::layout_map<1,0>
    \endcode
    The second template argument is still a \link gridtools::layout_map
    \endlink , but this time it indicates the mapping between data
    and processor grid. The data is still condidered in the user
    convention.
    Suppose the processor gris has size PIxPJ, and these may be the
    MPI communicator sizes (coords[0]=PI, coords[1]=PJ). Now, we want
    to say that the first dimension on data (first in the user
    convention, not int the increasing stride order) 'extends' to the
    computing gris, or that the first dimension in the data correspons
    to the first dimension in the computing grid. Let's consider a 2x1
    proc grid, and the first dimension of the data being the rows (i)
    and the second the column (j). In this case we are thinking to a
    distribution like this:
    \code
    >j>>
       ------
     v |0123|
     i |1234|  Proc 0,0
     v |2345|
     v |3456|
       ------
       >j>>
       ------
     v |4567|
     i |5678|  Proc 1,0
     v |6789|
     v |7890|
       ------
    \endcode
    In this case the map between data and the processor grid is:
    \code
    gridtools::layout_map<0,1>
    \endcode
    On the other hand, having specified
    \code
    gridtools::layout_map<1,0>
    \endcode
    for this map, would imply a layout/distribution like the following:
    \code
    >j>>                 >j>>
       ------               ------
     v |0123|             v |4567|
     i |1234|  Proc 0,0;  i |5678|  Proc 1,0
     v |2345|             v |6789|
     v |3456|             v |7890|
       ------               ------
    \endcode
    Where the second dimension in the data correspond to the fist
    dimension in the processor grid. Again, the data coordinates
    ordering is the one the user choose to be the logical order in the
    application, not the increasing stride order.
    A 3D example can be found in
    halo_exchange_3D_all.cpp
    The other template arguments are the type of the elements
    contained in the data arrays and the number of dimensions of
    data. The very last template argument is the architecture in which
    the application runs. This is used to specify if the data is in
    the main memory or in some other address space as GPUs. The
    description of the architectures is available in
    L3/include/gcl_arch.h .
    The pattern is constructed with the periodicity of the computing
    grid and the MPI cart communicator embedding the communicating
    processes. An example of contruction, for a 2D case corresponding
    to the previous example, is:
    \code
    typedef gridtools::halo_exchange_dynamic_ut<gridtools::layout_map<1,0>, gridtools::layout_map<1,0>, pair_t, 2,
   gridtools::gcl_cpu > pattern_type;
    pattern_type he(pattern_type::grid_type::period_type(true, true), CartComm);
    \endcode
    Subsequently the halos must be registered.
    \code
    he.add_halo<0>(minus0, plus0, begin0, end0, len0);
    he.add_halo<1>(minus1, plus1, begin1, end1, len1);
    \endcode
    The indices passed as template arguments indicate the order in
    which the dimensions of the data arrays have been thought by the
    user. So they are 0 is i and 1 is j in the previous example. The
    arguments specify the halo specification as in field_descriptor,
    halo_descriptor or \link MULTI_DIM_ACCESS \endlink (without the
    ordering implied there).
    When the registration is done a setup function must be called
    before running data exchange. the argument in the set up function
    is the maximum number of data array that the pattern will exchange
    in a single step. For instance we set that to 3 so that 4 data
    arrays can not be exchanged simultanously without triggering
    unpredictable results of, more likely, an access violation
    error. The code looks like:
    \code
    he.setup(3);
    he.pack(array0, array1, array3);
    he.start_exchange();
    he.wait();
    he.unpack(array0, array1, array3)
    \endcode
    An example of this code can be found in L2/test/test_halo_exchange_2D.h .
    \subsection L2-NALO_EX_LOWLEVEL Halo Exchange for regular grids at lower level
    The halo exchange for regular grids is implemented at lower level
    by classes \link gridtools::hndlr_descriptor_ut \endlink , \link
    gridtools::hndlr_dynamic_ut \endlink .
    The main idea behing the pattern is that users have a regular
    multidimensional array (called field sometimes here, but array is
    probably more approriate) that is split according to a
    multidimensional computing grid, where each processor actually is
    identified by a tuple of coordinates. The distributed computation,
    such as a stencil computation, accesses a certain number of
    elements that logically reside in portions of the arrays that are
    on other processes but are logically contigous in the big-array
    defining the problem in hand. To implement this, a area around
    each piece is filled with other portions from other processes and
    accessed seamlessly by the local process. This area, called ghost
    region or halo region, is what this pattern updates.
    In a simple 2D example when cyclic communication are set, the
    mapping between sending and receiving data are described in the
    folloing.
    \code
    ------------ ---------------
    |d ccccc dd| |ccdddddddd cc|
    | -------  | | ----------  |
    |b|aaaaa|bb| |a|bbbbbbbb|aa|
    |b|aaaaa|bb| |a|bbbbbbbb|aa|
    |b|aaaaa|bb| |a|bbbbbbbb|aa|
    | -------  | | ----------  |
    |d ccccc dd| |c dddddddd cc|
    ------------ ---------------
    ------------ ---------------
    |b aaaaa bb| |a bbbbbbbb aa|
    | -------  | | ----------  |
    |d|ccccc|dd| |c|dddddddd|cc|
    |d|ccccc|dd| |c|dddddddd|cc|
    |d|ccccc|dd| |c|dddddddd|cc|
    |d|ccccc|dd| |c|dddddddd|cc|
    |d|ccccc|dd| |c|dddddddd|cc|
    | -------  | | ----------  |
    |b aaaaa bb| |a bbbbbbbb aa|
    ------------ ---------------
    \endcode
    \subsection HE_STATIC The static version of the halo exchange
    There are currently to modalities to implement a Level 2 pattern
    using GCL. The first onecan be described as static and is
    implemented by \link gridtools::hndlr_descriptor_ut \endlink. In this
    case each process must describe the halos by using \link
    gridtools::halo_descriptor \endlink. The halo descrition process
    describes with 5 integer parameter each dimension of the arrays by
    increasing strides (the rationale behind this choice is discussed
    in \link MULTI_DIM_ACCESS \endlink ).
    More in detail, given a dimension it is described like this:
    \code
    |-----|------|---------------|---------|----|
    | pad0|minus |    length     | plus    |pad1|
    ^begin        ^end
    |               total_length                |
    \endcode
    In this way five numbers are needed to describe the whole
    structure, assuming the first index of pad0 is fixed to 0, and
    they are minus, begin, end, plus, and total_length. An array of
    halo_decsriptors decsribes a sub array of the full array. The
    dimensions must be specified in increasing order of stride
    size. So a typical C implementation of a 3D array would have to
    register the halos with the pattern starting with the last
    coordinate first and the first as last, a FORTRAN layout would
    refister the halos in the order they are spelled. The rationale
    behind this is to spell out the innermost loop of the most
    efficient scan of the data first, and register the other by
    decreasing performance.
    Before registering the halo, a pattern must be constructed as an object in the application. To do so
    \code
    typedef gridtools::Halo_Exchange_2D<PROC_GRID_TYPE> pattern_type;
    // This constructor can be used if an mpi cart communicator is already available
    gridtools::hndlr_descriptor_ut<char,2, pattern_type> hd(PROC_GRID_TYPE::period_type(false,true),MPI_COMMUNICATOR);
    // or
    // this constructor can be used if not
    gridtools::hndlr_descriptor_ut<char,2, pattern_type> hd(PROC_GRID_TYPE::period_type(false,true), NPROCS,
   MY_PROC_ID);
    \endcode

    The periodicity is specified in order of dimension in the
    computing gris, so the pattern will be periodic in the second
    dimension. computing grids are the same as the computing grids in
    Level 3 and the concepts are described in \link
    proc_grid_2D_concept \endlink and \link proc_grid_2D_concept
    \endlink .
    After the construction is done the registration can happen. First
    of all the array must be registered. the registration return an
    integer number to be used in the resistration of the halos for
    this array. The halo registration takes this number as first
    index, then the index of the dimension in the increasing stride
    order, then the 5 usual parameters.
    Multiple arrays can be registered with a single pattern with
    different sizes. At the end, whe the pattern is executed, all of
    them are exchanged in a single step.
    \code
    // a is the address of the first element of the data array
    I = hd.register_field(a);
    hd.register_halo(I, 0, 2, 1, 3, 6, 10);
    hd.register_halo(I, 1, 2, 1, 3, 6, 10);
    I = hd.register_field(b);
    hd.register_halo(I, 0, 3, 2, 3, 6, 10);
    hd.register_halo(I, 1, 2, 1, 3, 6, 10);
    I = hd.register_field(c);
    hd.register_halo(I, 0, 0, 2, 0, 6, 10);
    hd.register_halo(I, 1, 3, 2, 3, 6, 10);
    \endcode
    Before performing the data exchange the function the function
    \code
    hd.allocate_buffers();
    \endcode
    must be called (this is probably a bad name, it should be called
    setup() and it will change soon).  Subsequently the pattern can be
    executed in one of two ways. The first a synchronous call:
    \code
    hd.pack();

    hd.exchange();

    hd.unpack();
    \endcode
    Or otherwise as a split phase communication:
    \code
    hd.pack();

    hd.start_exchange();
    hd.wait();

    hd.unpack();
    \endcode
    In between start_exchange and wait other work can be performed,
    but the arrays conaining the data to be exchanged sould not be
    touched, even though the pack and unpack seems to save the user
    from semantic issues. Future implementation would probably remove
    the pack and unpack altogether.
    \subsection HE_DYNAMIC The dynamic version of the halo exchange
    The implementation is found in \link gridtools::hndlr_dynamic_ut
    \endlink . The difference with the previousis that the actual
    pointers to data are assumed to be unknown at pattern
    instantiation, so that they have to be passed to it just before
    the communication happens. While the construction has the same
    interface as in the previous section, the registration of the
    halos are different. Here is how it looks like. first of all, al
    the arrays passed to the pattern are assumed to have the same halo
    structure and sizes. So, after the pattern is constructed, the
    registration has the following structure, for a 3D example in this
    case:
    \code
    hd.halo.add_halo(2, H, H, H, DIM3-H-1, DIM3+padding3);
    hd.halo.add_halo(1, H, H, H, DIM2-H-1, DIM2+padding2);
    hd.halo.add_halo(0, H, H, H, DIM1-H-1, DIM1+padding1);
    hd.allocate_buffers(N);
    \endcode
    The example assumes the arrays are padded with some quantity, just
    for generality. The first parameter of add_halo is the index of a
    dimension in the usual increasing stride order (here the order of
    registration is reversed for purpose of illustration).
    The allocate_buffer function takes the maximum number of data
    arrays that will be passed to the exchange methods. In no occasion
    that number can be exceeded. If the compiler supports C++11
    features, then the maximum specifiable number is actually compiler
    dependent, otherwhise, the maximum number canno exceed the
    GCL_MAX_FIELDS parameter defined in \link
    gcl_parameters.hpp \endlink
    \section L2_ALL_TO_ALL All to all with halo specification
    The all to all pattern uses the same halo descriptors used
    previously in L2-HALO_EX halo exchange. The idea is that each
    process resister with the pattern the sub-arrays assigned to each
    specific destination and each specific source.
    For instance, if process 0 need to scatter a 2D array to the other
    processes in a 2D processing grid in a regular way, it executes
    something like the following code. First the pattern must be
    constructed. The constructor takes the processing grid.
    \code
    typedef gridtools::_2D_process_grid_t<gridtools::gcl_utils::boollist<2> > grid_type;
    grid_type pgrid(gridtools::gcl_utils::boollist<2>(true,true), gridtools::PROCS, gridtools::PID);
    gridtools::all_to_all_halo<int, grid_type> a2a(pgrid);
    \endcode
    Subsequently, the data to be sent and received must be specified.
    \code
    for(int i=0; i<PI; ++i) {
    for (int j=0; j<PJ; ++j) {
    crds[0] = i;
    crds[1] = j; // INCREASING STRIDES
    // INCREASING STRIDES
    send_block[1] = gridtools::halo_descriptor(0,0,j*N,(j+1)*N-1, PJ*N);
    send_block[0] = gridtools::halo_descriptor(0,0,i*N,(i+1)*N-1, N*PI);
    a2a.register_block_to(&dataout[0], send_block, crds);
    }
    }
    \endcode
    This time, to falicitate multidimensional implementation, the
    interface has choosen to use arrays or vectors of halo_descriptors
    instead of registering them one by one. The amount of code is not
    much saved from the user perspective, but from the library
    developer this has quite an impact. As the discussion about the
    interfaces proceeds, one choice over the other will probably
    prevail. send_block is, at this time an array of
    block_descriptors. The only requirements to this array is that it
    provide operator[] and a size() method to query array length.
    Also crds is an array with the coordinates of the receiving
    process. The only requirement is that it has to provide
    operator[].
    The first argument of the registration function is the pointer to
    the array containing the data to send.
    To implement a scatter, each other process, and 0, too, must
    declare where the data vas to be placed, and this happens with a
    call like the following one:
    \code
    recv_block[0] = gridtools::halo_descriptor(H,H,H,N+H-1, N+2*H);
    recv_block[1] = gridtools::halo_descriptor(H,H,H,N+H-1, N+2*H);
    a2a.register_block_from(&datain[0], recv_block, crds);
    \endcode
    Here recv_block is an array of halo_descriptors with size method,
    and H is the halo parameter around the receiving array.
    Now the pattern is ready to go, provided a call to a setup method:
    \code
    a2a.setup();
    a2a.start_exchange();
    a2a.wait();
    \endcode
 */

/** \page GCL_L1 GCL Level 1
 */

/** \page MULTI_DIM_ACCESS Increasing strides access and packing/unpacking of halos
    Here we assume that a D-dimensional array of sizes \f[ N_1\times N_2\times N_3 \ldots \times N_D \f]
    is such that in the first dimension the stride is 1, in the second it is N1, in the i-th it is
    \f[ N_1\cdot N_2\cdot \ldots \cdot N_{i-1}. \f]
    To linearized index to access element \f[(i_1, i_2,\ldots, i_D)\f] in this case is
    \f[\sum_{\ell=1}^D i_\ell \prod_{m=1}^{\ell-1}N_m \f] where the result of the product is 1 if ill-defined.
    We dscribe each dimension of the array as
    \code
    |-----|------|---------------|---------|----|
    | pad0|minus |    length     | plus    |pad1|
                  ^begin        ^end
    |               total_length                |
    \endcode
    Now suppose the D-dimensional array is distributed in blocks on a D-dimensional parallel computer,
    in which processors are identified by coordinates analogously to the elements of the array.
    Given a processor \f$P=(P_1,\ldots P_D)\f$ its immediate neighbors can be indicated by
    \f[p(\bar\eta)=(P_1+\eta_1,P_2+\eta_2,\ldots, P_D+\eta_D),\f] where \f$\eta_i \in \{-1,0,1\}.\f$
    Now, fixed \f$p(\bar\eta)\f$, let \f[S_i=\left\{\begin{array}{lr}\eta_i=0,& \mbox{end}_i-\mbox{being}_i+1\\
   \eta_i=1,&\mbox{plus}_i\\ \eta_i=-1,&\mbox{minus}_i\\ \end{array}\right.\f]
    The number of elements that need to be placed in the halo region for processor \f$P\f$ (indicated with
    plus and minus in the previous picture) is:
    \f[\prod_{i=1}^D S_i\f]
    Fixed \f$p(\bar\eta)\f$, If we define B_i to be the range of indices in the i-th dimension that corresponds
    to the elements of the halo in the direction of \f$p(\bar\eta)\f$, then to access these elements we need a loop-nest
    like
    \code
    for i1 in B1
      for i2 in B2
        .
         .
          for iD in BD
             access with the access formula given above
    \endcode
    We then define the ranges (inclusives):
    \f[B_i=\left\{\begin{array}{lr}\eta_i=0,& \mbox{begin}_i:\mbox{end}_i\\
                                   \eta_i=1,&\mbox{end}_i+1:\mbox{end}_i+\mbox{plus}_i\\
                                   \eta_i=-1,&\mbox{begin}_i-\mbox{minus}_i:\mbox{begin}_i-1\\
                  \end{array}\right.\f]
    The loop nest can be more efficient by reverting it, but this has only performance implications. The order
    must be consistent in sending and receiving sided for correctness. Remeber that these ranges are for the region
    outside the begin-end region. For the regions inside that region (the one that must be sent to the neighbors), the
    bound change a little. We address this in what follows.
    In fact the data to be taken from the inside for sending it to \f$p(\bar\eta)\f$ have to be treted sligthly
   differently.
    The sizes are
    \f[S'_i=\left\{\begin{array}{lr}\eta_i=0,& \mbox{end}_i-\mbox{being}_i+1\\ \eta_i=-1,&\mbox{plus}_i\\
   \eta_i=1,&\mbox{minus}_i\\ \end{array}\right.\f]
    While the sizes are the same, the bounds are slightly different:
    \f[B'_i=\left\{\begin{array}{lr}\eta_i=0,& \mbox{begin}_i:\mbox{end}_i\\
    \eta_i=-1,&\mbox{begin}_i:\mbox{begin}_i+\mbox{plus}_i-1\\
    \eta_i=1,&\mbox{end}_i-\mbox{minus}_i+1:\mbox{end}_i\\
    \end{array}\right.\f]
    For the interaction between data layout and computing grids, see
    \link GRIDS_INTERACTION \endlink
*/

/** \page GRIDS_INTERACTION Computing grids and data layout in GCL Level 2
    The dimensions Are registerd with the Level 2 Halo exchange
    pattern by increasing strides (see MULTI_DIM_ACCESS). The pattern
    exchanges data to neighbors which are also identified by
    coordinates (relative to the considered process).
    In this context, given a process \f[(p_0, p_1,\ldots p_D)\f] the
    neighbor \f[(p_0+1, p_1,\ldots p_D)\f] is logically the neighbor
    along the smallest stride, while \f[(p_0, p_1,\ldots p_D+1)\f]
    with the largest.
    So, when the data is shipped from a process to a neighbor, the
    same layout is assumed for the data and the computing grid.
    While this is completely transparent to the library, the
    application may need special attention to the logic. This can be
    seen in \link descriptors.hpp \endlink
    In this example the coordinates to the pattern are registered in
    inverse order with respect the tuple ordering (i,j,k). that is k
    has the smallest stride while i has the largest. To fill the data
    correctly and check correctness, the coordinates and sizes
    returned by the computing grid are assuming (k,j,i) coordinates,
    since the first has the smallest stride. So, to be clear, the
    coordinates are picked up in the inverse order, or we can say in
    the same order as the data layout have been passed to the library.
*/

/** \page Concepts Concepts
    Concepts are not classes present in the library. They are
    decsription of interfaces. If the user of the library implements a
    class according to a concept, that class can be used in the
    library as any regular library class. For instance, users can
    define their own 3D processor grids and use them with the halo
    eachange communication patterns provided by the library.
    \section proc_grid_concept Processor Grid Concept
    A process grid is a class that provides the interfaces of the
    generic class (concept) \link proc_grid_2D_concept \endlink for 2D
    case and \link proc_grid_3D_concept \endlink for 3D case
    \section utils_concepts Utils Concepts
    \link boollist_concept \endlink
*/

/** @} */
