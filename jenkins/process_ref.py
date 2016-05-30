#from graph_tool.all import *
import json
import argparse
import subprocess
import re,sys
import math
import os
import socket
import numpy as np
import matplotlib
matplotlib.use('SVG')
from matplotlib import rc
import matplotlib.pyplot as plt
import copy
import os.path
import datetime

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


def run_and_extract_times(executable, host, sizes, filter_=None, stella_format = None, verbosity=False):

    cmd=''
    cmd = ". "+os.getcwd()+"/env_"+host+".sh; "

    if stella_format:
        cmd = cmd + executable +' --ie ' + str(sizes[0]) + ' --je ' + str(sizes[1]) + ' --ke ' + str(sizes[2])
    else:
        cmd = cmd + executable +' ' + str(sizes[0]) + ' ' + str(sizes[1]) + ' ' + str(sizes[2]) + ' 10'
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

    avg_time = avg_time / float(nrep)
    rms=0
    for t in times:
        rms = rms + (t-avg_time)*(t-avg_time)
    rms = math.sqrt(rms) / nrep

    return (avg_time,rms)

class Plotter:
    def __init__(self, reference_timers, gridtools_timers, stella_timers, config, branch_name):
        self.reference_timers_ = reference_timers
        self.gridtools_timers_ = gridtools_timers
        self.stella_timers_ = stella_timers
        self.config_ = config
        self.branch_name_ = branch_name

        self.stella_avg_times_ = {}
        self.stella_err_ = {}
 
        self.gridtools_avg_times_ = {}
        self.gridtools_err_ = {}
        self.reference_avg_times_ = {}
        self.reference_err_ = {}

        self.labels_ = {}

    def plot(self, filename, title, xtick_labels, y1, y1_err, label1, y2, y2_err, label2):
    
        n_groups = len(y1)
        index = np.arange(n_groups)
        bar_width = 0.25

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        y1_bar = plt.bar(index , y1, yerr = y1_err, width=bar_width,
              alpha=opacity,
              color='r',
              label=label1)

        y1_errbar = plt.errorbar(index +bar_width*0.5, y1, yerr = y1_err, color='r', ls='none')

        y2_bar = plt.bar(index + bar_width, y2, yerr= y2_err, width=bar_width,
               alpha=opacity,
               color='b',
               label=label2)
        y2_errbar = plt.errorbar(index + bar_width*1.5, y2, yerr= y2_err, color='b', ls='none')


        plt.xlabel('Stencil Name')
        plt.ylabel('Stencil time (s)')
        plt.title(title)
        plt.xticks(index + bar_width*1.5, xtick_labels, rotation=90, fontsize='xx-small')
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='both', which='minor', labelsize=6)
        plt.legend(prop={'size':6})

        plt.tight_layout()
        plt.savefig(filename, format="svg")
      
    def plot_titlepage(self, filename):
       x,y = 0.1,0.4

       fig = plt.figure()
       ax = fig.add_subplot(111)
       rc('font',**{'size':24 })
       rc('text')
       y += 0.3
       ax.text(x,y,'Performance Results for Branch: ')
       ax.text(x+0.05, y-0.2, self.branch_name_)
       ax.text(x+0.05, y-0.4, str(datetime.datetime.now()))

#       ax = plt.axes()
       ax.xaxis.set_visible(False)
       ax.yaxis.set_visible(False)
       plt.savefig(filename, format="svg")



    def plot_results(self):

        self.extract_metrics()

        if not os.path.exists("perf_vs_stella"):
            os.makedirs("perf_vs_stella")

        for astencil in self.stella_avg_times_:
            for adomain in self.stella_avg_times_[astencil]:
                fig, ax = plt.subplots()

                stella_times = [a for a in self.stella_avg_times_[astencil][adomain] ]
                gridtools_times = self.gridtools_avg_times_[astencil][adomain]
                stella_err = [err for err in self.stella_err_[astencil][adomain] ]
                gridtools_err = self.gridtools_err_[astencil][adomain]
                labels = self.labels_[astencil][adomain]
                
                self.plot("perf_vs_stella/plot_"+astencil+"_"+adomain+".svg", astencil, labels, stella_times, stella_err, "stella", gridtools_times, gridtools_err, "gridtools")

        if not os.path.exists("perf_vs_reference"):
            os.makedirs("perf_vs_reference")

        if not os.path.exists("aa_title"):
            os.makedirs("aa_title")
        self.plot_titlepage("aa_title/aaa_titlepage.svg")

        for astencil in self.gridtools_avg_times_:
            for adomain in self.gridtools_avg_times_[astencil]:
                fig, ax = plt.subplots()

                gridtools_times = self.gridtools_avg_times_[astencil][adomain]
                reference_times = self.reference_avg_times_[astencil][adomain]
                gridtools_err = self.gridtools_err_[astencil][adomain]
                reference_err = self.reference_err_[astencil][adomain]
                labels = self.labels_[astencil][adomain]
                
                self.plot("perf_vs_reference/plot_"+astencil+"_"+adomain+".svg", astencil + ' ' + adomain, labels, gridtools_times, gridtools_err, "gridtools", reference_times, reference_err, "reference")

    def stella_has_stencil(self, stencil_name):
        return self.stella_timers_.has_key(stencil_name)

    def extract_metrics(self):

        for astencil in self.gridtools_timers_:
            if self.stella_has_stencil(astencil):
                self.stella_avg_times_[astencil] = {}
                self.stella_err_[astencil] = {}
            self.gridtools_avg_times_[astencil] = {}
            self.labels_[astencil] = {}
            self.gridtools_err_[astencil] = {}
            self.reference_avg_times_[astencil] = {}       
            self.reference_err_[astencil] = {}
 
            gridtools_this_stencil_data = self.gridtools_timers_[astencil][self.config_.target_][self.config_.prec_][self.config_.std_]

            for athread_num in gridtools_this_stencil_data:
                for adomain in gridtools_this_stencil_data[athread_num]:
                    if not self.gridtools_avg_times_[astencil].has_key(adomain):
                        if self.stella_has_stencil(astencil):
	                    self.stella_avg_times_[astencil][adomain] = []
                            self.stella_err_[astencil][adomain] = []
                        self.gridtools_avg_times_[astencil][adomain] = []
                        self.labels_[astencil][adomain] = []
                        self.gridtools_err_[astencil][adomain] = []
                        self.reference_avg_times_[astencil][adomain] = []
                        self.reference_err_[astencil][adomain] = []

                    if self.stella_has_stencil(astencil):
                        self.stella_avg_times_[astencil][adomain].append( self.stella_timers_[astencil][athread_num][adomain][0])
                        self.stella_err_[astencil][adomain].append( self.stella_timers_[astencil][athread_num][adomain][1])

                    self.gridtools_avg_times_[astencil][adomain].append(
                        self.gridtools_timers_[astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['time'])

                    self.gridtools_err_[astencil][adomain].append(
                                            self.gridtools_timers_[astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['rms'])
                    self.reference_avg_times_[astencil][adomain].append(
                        self.reference_timers_[astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['time'])

                    self.reference_err_[astencil][adomain].append(
                                            self.reference_timers_[astencil][self.config_.target_][self.config_.prec_][self.config_.std_][athread_num][adomain]['rms'])

                    self.labels_[astencil][adomain].append(str(athread_num))

        return

def create_dict(adict, props):
    if not props: return

    current_prop = props.pop()
    if not adict.has_key(current_prop):
        adict[current_prop]={}

    create_dict(adict[current_prop], props)

class Config:
    def __init__(self, target, prec, std, update, check):
        self.target_ = target
        self.prec_ = prec
        self.std_ = std
        self.update_ = update
        self.check_ = check

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
    parser.add_argument('-c', action='store_true', help='check results and validate against a reference')
    parser.add_argument('-u', action='store_true', help='update reference performance results')
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

    stella_path = None
    gridtools_path = None
    if args.stella_path:
        stella_path = args.stella_path[0]
    gridtools_path = args.p[0]

    if stella_path and not os.path.exists(stella_path):
        parser.error('STELLA build path '+stella_path+' does not exists')
    if not os.path.exists(gridtools_path):
        parser.error('GridTools build path '+gridtools_path+' does not exists')

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

    stella_exec = None
    if args.stella_path:
        stella_exec = stella_path + '/StandaloneStencils'+stella_suffix
    update = False
    check = False
    if args.u:
        update = True
    if args.c:
        check = True

    prec = args.prec[0]
    if prec != 'float' and prec != 'double':
        parser.error('-prec must be set to float or double')
    do_plot = False
    if args.plot:
        do_plot = True

    verbose=False
    if args.v:
        verbose=True

    host = filter(lambda x: x.isalpha(), socket.gethostname())

    if not re.match('greina', host) and not re.match('kesch', host):
        sys.exit('WARNING: host '+host+' not known. Not loading any environment')
    
    if re.match('greina', host):
        host='greina'
    elif re.match('kesch', host):
        host='kesch'

    config = Config(target,prec,std, update, check)

    f = open(args.json_file,'r')
    decode = json.load(f)

    failed = False
    nrep=3

    copy_ref = copy.deepcopy(decode)
    stella_timers = {}

    for stencil_name in decode[host]['stencils']:
        print('CHECKING :', stencil_name)
        skip=True
        for filter_stencil in filter_stencils:
            filter_re = re.compile(filter_stencil)

            if filter_re.search(stencil_name):
                skip=False
        if filter_stencils and skip: 
            print('Skipping ',stencil_name)
            continue

        stencil_data = decode[host]['stencils'][stencil_name]
        executable = gridtools_path+'/'+stencil_data['exec']+'_'+target_suff

        stella_filter = stencil_data['stella_filter']

        for thread in stencil_data[target][prec][std]: 
            domain_data = stencil_data[target][prec][std][thread]
            for data in domain_data:
                sizes = data.split('x')
                exp_time = domain_data[data]['time']
                
                timers_gridtools = run_and_extract_times(executable, host, sizes, verbosity=verbose)

                if stella_exec and stella_filter:
                    create_dict(stella_timers, [data, thread, stencil_name] )
                    stella_timers[stencil_name][thread][data] = run_and_extract_times(stella_exec, host, sizes, stella_filter, stella_format=True, verbosity=verbose)

                copy_ref[host]['stencils'][stencil_name][target][prec][std][thread][data]['time'] = timers_gridtools[0]
                copy_ref[host]['stencils'][stencil_name][target][prec][std][thread][data]['rms'] = timers_gridtools[1]

                error = math.fabs(float(timers_gridtools[0]) - float(exp_time)) / (float(exp_time)+1e-20)
                if config.check_ and error > tolerance:
                    print('Error in conf ['+stencil_name+','+data+','+prec+','+target+','+std+','+thread+'] : exp_time -> '+ str(exp_time) + '; comp time -> '+ 
                        str(timers_gridtools[0])+'. Error = '+ str(error*100)+'%')
                    failed = True

    if config.update_:
        outputfilename=args.json_file +'.out'
        fw = open(outputfilename,'w')
        fw.write(json.dumps(copy_ref,  indent=4, separators=(',', ': ')) )
        fw.close()
        print("Updated reference file",outputfilename)

    if do_plot:
        branch_name=subprocess.check_output('git branch --contains `git rev-parse HEAD` -r', stderr=subprocess.STDOUT, shell=True)
        plotter = Plotter(decode[host]['stencils'], copy_ref[host]['stencils'], stella_timers, config, branch_name)
        plotter.plot_results()

    if not failed:
        print('[OK]')
    else:
        print('[FAILED]')
    sys.exit(int(failed))


