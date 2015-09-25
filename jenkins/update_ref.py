#from graph_tool.all import *
import json
import argparse
import subprocess
import re,sys
import math

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


if __name__ == "__main__":

    tolerance = 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="filename containing the stencils json report")
    parser.add_argument('--filter',nargs=1, type=str, help='filter only stencils that matches the pattern')
    parser.add_argument('-p',nargs=1, type=str, help='filter only stencils that matches the pattern')
    parser.add_argument('--target', nargs=1, type=str, help='cpu || gpu')
    parser.add_argument('--std', nargs=1, type=str, help='C++ standard')

    filter_stencils = None    
    args = parser.parse_args()
    if args.filter:
        filter_stencils = args.filter
    
    if args.p:
        path=args.p[0]
    else:
        parser.error('-p path should be specified')

    if not args.target:
        parser.error('--target should be specified')

    if not args.std:
        parser.error('--std should be specified')

    target = args.target[0]
    if target == 'gpu':
        target_suff = "cuda"
    elif target == 'cpu':
        target_suff = "block"

    std = args.std[0]
    if std != "cxx11" and std != "cxx03":
        parser.error('--std should be set to cxx11 or cxx03')

    f = open(args.json_file,'r')
    decode = json.load(f)

    copy_ref = decode
    nrep = 3

    for stencil_name in decode['stencils']:
        stencil_data = decode['stencils'][stencil_name]
        executable = path+'/'+stencil_data['exec']+'_'+target_suff

        for thread in stencil_data[target][std]: 
            domain_data = stencil_data[target][std][thread]
            for data in domain_data:
                sizes = data.split('x')
                exp_time = domain_data[data]
                cmd = executable +' ' + str(sizes[0]) + ' ' + str(sizes[1]) + ' ' + str(sizes[2])
                if target == 'cpu':
                    cmd = 'export OMP_NUM_THREADS='+thread+'; '+cmd
                avg_time=0
                times=[]
                for i in range(nrep):
                    try:
                        output=subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                        m = re.search('.*\[s\]\s*(\d+(\.\d*)?|\.\d+)',output)
                        if m:
                            time = m.group(1)
                            avg_time = avg_time + float(time)
                            times.append(float(time))
                    except subprocess.CalledProcessError, e:
                        sys.exit('Command called raised error:\n'+e.output)

                avg_time = avg_time / nrep
                rms=0
                for t in times:
                    rms = rms + (t-avg_time)*(t-avg_time)
                rms = math.sqrt(rms) / nrep

                copy_ref['stencils'][stencil_name][target][std][thread][data]['time'] = avg_time
                copy_ref['stencils'][stencil_name][target][std][thread][data]['rms'] = rms

    fw = open("write",'w')
    fw.write(json.dumps(copy_ref,  indent=4, separators=(',', ': ')) )
    fw.close()


     #   print(domain_data)

