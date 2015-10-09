#from graph_tool.all import *
import json
import argparse
import subprocess
import re,sys
import math
import os
import socket
import numpy as np
import matplotlib.pyplot as plt

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


def run_and_extract_times(executable, sizes, filter_=None, stella_format = None, verbosity=False):

    machine = filter(lambda x: x.isalpha(), socket.gethostname())

    cmd=''
    if machine != 'greina':
        print('WARNING: machine '+machine+' not known. Not loading any environment')
    else:
        cmd = ". "+os.getcwd()+"/env_"+machine+".sh; "

    if stella_format:
        cmd = cmd + executable +' --ie ' + str(sizes[0]) + ' --je ' + str(sizes[1]) + ' --ke ' + str(sizes[2])
    else:
        cmd = cmd + executable +' ' + str(sizes[0]) + ' ' + str(sizes[1]) + ' ' + str(sizes[2])
    if filter_:
        cmd = cmd + ' ' + filter_
    if target == 'cpu':
        nthreads = re.sub('thread','',thread)
        cmd = 'export OMP_NUM_THREADS='+nthreads+'; '+cmd

    avg_time = 0

    times = []
    for i in range(nrep):
        try:
            if verbosity:
                print(cmd)
            output=subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
            if verbosity:
                print(output)
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

    return (avg_time,rms)

class Plotter:
    def __init__(self, gridtools_timers, stella_timers, config):
        self.gridtools_timers_ = gridtools_timers
        self.stella_timers_ = stella_timers
        self.config_ = config

        self.stella_avg_times_ = {}
        self.stella_err_ = {}
 
        self.gridtools_avg_times_ = {}
        self.gridtools_err_ = {}

        self.labels_ = {}


    def plot_results(self):

        self.extract_metrics()

        for astencil in self.stella_avg_times_:
            for adomain in self.stella_avg_times_[astencil]:
                fig, ax = plt.subplots()

                stella_times = [a/10.0 for a in self.stella_avg_times_[astencil][adomain] ]
                gridtools_times = self.gridtools_avg_times_[astencil][adomain]
                stella_err = [err/10.0 for err in self.stella_err_[astencil][adomain] ]
                gridtools_err = self.gridtools_err_[astencil][adomain]
                labels = self.labels_[astencil][adomain]
                

                n_groups = len(stella_times)
                index = np.arange(n_groups)
                bar_width = 0.25

                opacity = 0.4
                error_config = {'ecolor': '0.3'}

                stella_bar = plt.bar(index , stella_times, yerr = stella_err, width=bar_width,
                     alpha=opacity,
                     color='r',
                     label='stella')

                stella_errbar = plt.errorbar(index +bar_width*0.5, stella_times, yerr = stella_err, color='r', ls='none')


                gridtools_bar = plt.bar(index + bar_width, gridtools_times, yerr= gridtools_err, width=bar_width,
                     alpha=opacity,
                     color='b',
                     label='gridtools')
                gridtools_errbar = plt.errorbar(index + bar_width*1.5, gridtools_times, yerr= gridtools_err, color='b', ls='none')


                plt.xlabel('Stencil Name')
                plt.ylabel('Stencil time (s)')
                plt.title(astencil)
                plt.xticks(index + bar_width*1.5, labels, rotation=90, fontsize='xx-small')
                plt.legend()

                plt.tight_layout()
                plt.savefig("plot_"+astencil+"_"+adomain+".svg", format="svg")


    def extract_metrics(self):

        for astencil in self.stella_timers_:
            self.stella_avg_times_[astencil] = {}
            self.gridtools_avg_times_[astencil] = {}
            self.labels_[astencil] = {}
            self.stella_err_[astencil] = {}
            self.gridtools_err_[astencil] = {}
        
            for athread_num in self.stella_timers_[astencil]:
                for adomain in self.stella_timers_[astencil][athread_num]:
                    if not self.stella_avg_times_[astencil].has_key(adomain):
                        self.stella_avg_times_[astencil][adomain] = []
                        self.gridtools_avg_times_[astencil][adomain] = []
                        self.labels_[astencil][adomain] = []
                        self.stella_err_[astencil][adomain] = []
                        self.gridtools_err_[astencil][adomain] = []

                    self.stella_avg_times_[astencil][adomain].append( self.stella_timers_[astencil][athread_num][adomain][0])
                    self.gridtools_avg_times_[astencil][adomain].append(
                        self.gridtools_timers_['stencils'][astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['time'])

                    self.stella_err_[astencil][adomain].append( self.stella_timers_[astencil][athread_num][adomain][1])
                    self.gridtools_err_[astencil][adomain].append(
                                            self.gridtools_timers_['stencils'][astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['rms'])
                    self.labels_[astencil][adomain].append(str(athread_num))
            print "HH", astencil

        return

def create_dict(adict, props):
    if not props: return

    current_prop = props.pop()
    if not adict.has_key(current_prop):
        adict[current_prop]={}

    create_dict(adict[current_prop], props)

class Config:
    def __init__(self, target, prec, std):
        self.target_ = target
        self.prec_ = prec
        self.std_ = std

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
    parser.add_argument('--plot', action='store_true', help='plot the comparison timings')
    parser.add_argument('--stella_path', nargs=1, type=str, help='path to stella installation dir')
    parser.add_argument('-v',action='store_true', help='verbosity')


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
    stella_suffix=""
    if target == 'gpu':
        target_suff = "cuda"
        stella_suffix = "CUDA"
    elif target == 'cpu':
        target_suff = "block"
    else:
        parser.error('wrong value for --target')

    std = args.std[0]
    if std != "cxx11" and std != "cxx03":
        parser.error('--std should be set to cxx11 or cxx03')

    if not args.m:
        parser.error('mode -m should be specified')

    stella_exec = None
    if args.stella_path:
        stella_exec = args.stella_path[0] + '/StandaloneStencils'+stella_suffix

    mode = args.m[0]
    if mode != 'u' and mode != 'c':
        parser.error('-m must be set to c or u')

    prec = args.prec[0]
    if prec != 'float' and prec != 'double':
        parser.error('-prec must be set to float or double')
    do_plot = False
    if args.plot:
        do_plot = True

    verbose=False
    if args.v:
        verbose=True
    config = Config(target,prec,std)

    f = open(args.json_file,'r')
    decode = json.load(f)

    error = False
    nrep=3

    copy_ref = decode
    stella_timers = {}

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

        stella_filter = stencil_data['stella_filter']

        for thread in stencil_data[target][prec][std]: 
            domain_data = stencil_data[target][prec][std][thread]
            for data in domain_data:
                sizes = data.split('x')
                exp_time = domain_data[data]['time']
                
                timers_gridtools = run_and_extract_times(executable, sizes, verbosity=verbose)

                if stella_exec and stella_filter:
                    create_dict(stella_timers, [data, thread, stencil_name] )
                    stella_timers[stencil_name][thread][data] = run_and_extract_times(stella_exec, sizes, stella_filter, stella_format=True, verbosity=verbose)

                copy_ref['stencils'][stencil_name][target][prec][std][thread][data]['time'] = timers_gridtools[0]
                copy_ref['stencils'][stencil_name][target][prec][std][thread][data]['rms'] = timers_gridtools[1]

                error = math.fabs(float(timers_gridtools[0]) - float(exp_time)) / (float(exp_time)+1e-20)
                if mode == 'c' and error > tolerance:
                    print('Error in conf ['+data+','+prec+','+target+','+std+','+thread+'] : exp_time -> '+ str(exp_time) + '; comp time -> '+ 
                        str(timers_gridtools[0])+'. Error = '+ str(error*100)+'%')
                    error = True

    if mode == 'u':
        fw = open(args.json_file +'.out','w')
        fw.write(json.dumps(copy_ref,  indent=4, separators=(',', ': ')) )
        fw.close()

    if do_plot:
        plotter = Plotter(copy_ref, stella_timers, config)
        plotter.plot_results()

    if not error:
        print('[OK]')
    else:
        print('[FAILED]')
    sys.exit(int(error))


