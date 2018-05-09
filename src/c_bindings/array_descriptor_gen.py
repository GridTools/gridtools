#!/usr/bin/env python3
import textwrap

def print_copyright():
    print("""\
    ! GridTools Libraries !
    ! Copyright (c) 2017, ETH Zurich and MeteoSwiss
    ! All rights reserved.
    !
    ! Redistribution and use in source and binary forms, with or without
    ! modification, are permitted provided that the following conditions are
    ! met:
    !
    ! 1. Redistributions of source code must retain the above copyright
    ! notice, this list of conditions and the following disclaimer.
    !
    ! 2. Redistributions in binary form must reproduce the above copyright
    ! notice, this list of conditions and the following disclaimer in the
    ! documentation and/or other materials provided with the distribution.
    !
    ! 3. Neither the name of the copyright holder nor the names of its
    ! contributors may be used to endorse or promote products derived from
    ! this software without specific prior written permission.
    !
    ! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    ! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    ! A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    ! HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    ! SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    ! LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    ! DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    ! THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    ! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    ! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    !
    ! For information: http://eth-cscs.github.io/gridtools/
""")

def print_indented(code, N):
    indented = textwrap.indent(code, ''.join(['    '] * N))
    print('\n'.join(map(lambda l : l.rstrip(), indented.split('\n'))))

def print_gt_fortran_array_descriptor(indent=0):
    print_indented("""\
type, bind(c), public :: gt_fortran_array_descriptor
    integer(c_int) :: type
    integer(c_int) :: rank
    integer(c_int), dimension(7) :: dims
    type(c_ptr) :: data
    ! TODO: add support for strides, bounds end type gt_fortran_array_descriptor
end type gt_fortran_array_descriptor""", indent)

def print_all_fill_type_info(types, indent=0):
    for index, typename in types.items():
        print_indented("""\
subroutine fill_type_info""" + str(index) + """(dummy, descriptor)
    """ + typename + """, target :: dummy
    type(gt_fortran_array_descriptor) :: descriptor

    descriptor%type = """ + str(index) + """
end subroutine
""", indent)

def print_fill_type_info_interface(types, indent=0):
    print_indented("interface fill_type_info", indent)
    for index in types.keys():
        print_indented("    procedure fill_type_info" + str(index), indent)
    print_indented("end interface", indent)

def print_all_fill_array_dimensions(ranks, indent=0):
    for rank in ranks:
        print_indented("""\
subroutine fill_array_dimensions""" + str(rank) + """(array, descriptor)
    type(*), dimension(""" + ','.join([':'] * rank) + """), target :: array
    type(gt_fortran_array_descriptor) :: descriptor

    descriptor%rank = """ + str(rank) + """
    descriptor%dims = reshape(shape(array), &
    shape(descriptor%dims), (/0/))
end subroutine""", indent)

def print_fill_array_dimensions_interface(ranks, indent=0):
    print_indented("interface fill_array_dimensions", indent)
    for rank in ranks:
        print_indented("    procedure fill_array_dimensions" + str(rank), indent)
    print_indented("end interface", indent)

def print_fill_common_info(indent=0):
    print_indented("""\
subroutine fill_common_info(array, descriptor)
    type(*), dimension(*), target :: array
    type(gt_fortran_array_descriptor) :: descriptor

    descriptor%data = c_loc(array)

    print *, "rank: ", descriptor%rank
    print *, "dims: ", descriptor%dims
    print *, "type: ", descriptor%type
end subroutine""", indent)

def print_all_create_array_descriptor(types, ranks, indent=0):
    wrapper = textwrap.TextWrapper(break_long_words=False, subsequent_indent=''.join(' ' * 4 * (indent+1)))

    for rank in ranks:
        for index, typename in types.items():
            fname = "create_array_descriptor" + str(rank) + "_" + str(index)
            fill_type_info_call = "call fill_type_info(array(" \
                + ', '.join(map(lambda dim : "lbound(array, " + str(dim+1) + ")", range(rank))) \
                + "), descriptor)"
            fill_type_info_call = ' &\n'.join(wrapper.wrap(fill_type_info_call))
            print_indented("""
function create_array_descriptor""" + str(rank) + "_" + str(index) + """(array) result (descriptor)
    """ + typename + """, dimension(""" + ','.join([':'] * rank)+ """), target :: array
    type(gt_fortran_array_descriptor) :: descriptor

    """ + fill_type_info_call + """
    call fill_array_dimensions(array, descriptor)
    call fill_common_info(array, descriptor)

end function""", indent)

def print_create_array_descriptor_interface(types, ranks, indent=0):
    print_indented("interface create_array_descriptor", indent)
    for rank in ranks:
        for index in types.keys():
            print_indented("    procedure create_array_descriptor" + str(rank) + "_" + str(index), indent)
    print_indented("end interface", indent)

types = {1: 'real(c_float)',
         2: 'real(c_double)',
         3: 'integer(c_int)',
         4: 'integer(c_long)',
         5: 'logical(c_bool)'}
ranks = range(1, 8)

print_copyright()
print_indented("""module array_descriptor
    use iso_c_binding
    implicit none

    private
    public :: create_array_descriptor
    """, 0)
print_gt_fortran_array_descriptor(1)
print()
print_fill_type_info_interface(types, 1)
print()
print_fill_array_dimensions_interface(ranks, 1)
print()
print_create_array_descriptor_interface(types, ranks, 1)
print()
print()
print_indented("contains", 0)
print_fill_common_info(1)
print_all_fill_type_info(types, 1)
print_all_fill_array_dimensions(ranks, 1)
print_all_create_array_descriptor(types, ranks, 1)
print_indented("end module", 0)
