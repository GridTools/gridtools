#from graph_tool.all import *
import json
import argparse
import subprocess
import re,sys
import math
import os

def check_output(*popenargs, **kwargs):
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = subprocess.CalledProcessError(retcode, cmd)
        error.output = output
        print 'Error in command -> ',output
        raise error
    return output

try: subprocess.check_output
except: subprocess.check_output = check_output

"""

"""
if __name__ == "__main__":

    tolerance = 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="filename containing the stencils json report")
    parser.add_argument('--filter',nargs=1, type=str, help='filter only stencils that matches the pattern. Comma separated list of python regular expressions')
    parser.add_argument('-p',nargs=1, type=str, help='base path to build executables')
    parser.add_argument('--target', nargs=1, type=str, help='cpu || gpu')
    parser.add_argument('--std', nargs=1, type=str, help='C++ standard')
    parser.add_argument('--prec', nargs=1, type=str, help='floating point precision')
    parser.add_argument('-m', nargs=1, type=str, help='Mode: u (update reference), c (check reference)')

    filter_stencils = [] 
    args = parser.parse_args()
    if args.filter:
        filter_stencils = args.filter[0].split(',')
    if args.p:
        path=args.p[0]
    else:
        parser.error('-p path should be specified')

    if not args.target:
        parser.error('--target should be specified')
    if not args.std:
        parser.error('--std should be specified')
    if not args.prec:
        parser.error('--prec should be specified')

    target = args.target[0]
    if target == 'gpu':
        target_suff = "cuda"
    elif target == 'cpu':
        target_suff = "block"

    std = args.std[0]
    if std != "cxx11" and std != "cxx03":
        parser.error('--std should be set to cxx11 or cxx03')

    if not args.m:
        parser.error('mode -m should be specified')

    mode = args.m[0]
    if mode != 'u' and mode != 'c':
        parser.error('-m must be set to c or u')

    prec = args.prec[0]
    if prec != 'float' and prec != 'double':
        parser.error('-prec must be set to float or double')

    f = open(args.json_file,'r')
    decode = json.load(f)

    result = True
    nrep=3

    copy_ref = decode

    for stencil_name in decode['stencils']:
        print('CHECKING :', stencil_name)
        skip=True
        for filter_stencil in filter_stencils:
            filter_re = re.compile(filter_stencil)

            if filter_re.search(stencil_name):
                skip=False
        if filter_stencils and skip: 
            print('Skipping ',stencil_name)
            continue

        stencil_data = decode['stencils'][stencil_name]
        executable = path+'/'+stencil_data['exec']+'_'+target_suff

        for thread in stencil_data[target][prec][std]: 
            domain_data = stencil_data[target][prec][std][thread]
            for data in domain_data:
                sizes = data.split('x')
                exp_time = domain_data[data]['time']
                cmd = ". "+os.getcwd()+"/env.sh; " + executable +' ' + str(sizes[0]) + ' ' + str(sizes[1]) + ' ' + str(sizes[2])
                if target == 'cpu':
                    nthreads = re.sub('thread','',thread)
                    cmd = 'export OMP_NUM_THREADS='+nthreads+'; '+cmd

                avg_time = 0
                times = []
                for i in range(nrep):
                    try:
                        output=subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                        m = re.search('.*\[s\]\s*(\d+(\.\d*)?|\.\d+)',output)
                        if m:
                            extracted_time =  m.group(1)
                            avg_time = avg_time + float(extracted_time)
                            times.append(float(extracted_time))
                        else:
                            sys.exit('Problem found extracting timings')

                    except subprocess.CalledProcessError, e:
                        sys.exit('Command called raised error:\n'+e.output)

                avg_time = avg_time / 3.0
                rms=0
                for t in times:
                    rms = rms + (t-avg_time)*(t-avg_time)
                rms = math.sqrt(rms) / nrep


                copy_ref['stencils'][stencil_name][target][prec][std][thread][data]['time'] = avg_time
                copy_ref['stencils'][stencil_name][target][prec][std][thread][data]['rms'] = rms

                error = math.fabs(float(extracted_time) - float(exp_time)) / (float(exp_time)+1e-20)
                if mode == 'c' and error > tolerance:
                    print('Error in conf ['+data+','+prec+','+target+','+std+','+thread+'] : exp_time -> '+ str(exp_time) + '; comp time -> '+ extracted_time+'. Error = '+ str(error*100)+'%')
                    result = False

    if mode == 'u':
        fw = open(args.json_file +'.out','w')
        fw.write(json.dumps(copy_ref,  indent=4, separators=(',', ': ')) )
        fw.close()

    if result:
        print('[OK]')
    else:
        print('[FAILED]')
    sys.exit(result)

     #   print(domain_data)

