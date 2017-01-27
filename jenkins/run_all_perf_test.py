#/usr/bin/python
from itertools import product
import subprocess
import os.path
import os
import json
import shutil
import sys
import argparse
import distutils.dir_util as dutil

def build_path(host, jplan, target, prec, std):
    path="/scratch/jenkins/workspace/"+jplan+"/build_type/release/compiler/gcc/label/"+host+"/mpi/MPI/"
    if jplan == "GridTools":
        path=path+"/python/python_off"
    
    path=path+"/real_type/"+prec+"/std/"+std+"/target/"+target+"/build/"   

    return path

def build_outdir(gtype, target,prec,std):
    return gtype+'_'+target+'_'+prec+'_'+std

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--jplan',nargs=1, type=str, help='JENKINS plan')
    parser.add_argument('--gtype', nargs=1, type=str, help='Grid Type') 
    parser.add_argument('--std',nargs=1, type=str, help='list of stds to run')
    parser.add_argument('--prec',nargs=1, type=str, help='precision')

    args = parser.parse_args()
    if not args.jplan:
        parser.error('--jplan should be specified')
    else:
        jplan=args.jplan[0]

    plan_options=['GridTools','GridTools_strgrid_PR', 'GridTools_icgrid_PR']
    if jplan not in plan_options:
        parser.error('--jplan must be in', plan_options)

    if not args.gtype:
        parser.error('--gtype must be specified')
    else:
        gtype=args.gtype[0]

    if gtype != 'icgrid' and gtype != 'strgrid':
        parser.error('--gtype must be icgrid || strgrid')

    if gtype == 'icgrid':
        json_file = 'stencils_icgrid.json'
    else:
        json_file = 'stencils_strgrid.json'

    if args.std:
        stds=args.std[0].split(',')
    else:
        stds='cxx11'

    if args.prec:
        precs=args.prec[0].split(',')
    else:
        precs=('float', 'double')

    json_file_out = json_file+'.out'
    targets=('gpu','cpu')
    
    print('Running for confs: ', stds, targets, precs)

    commit_hash=None
    for target, prec, std in product(targets, precs, stds):
        path=build_path("kesch",jplan, target, prec, std)
        gitrev_cmd='git rev-parse  HEAD '+path 
        gitrev_out=subprocess.Popen(gitrev_cmd, shell=True, stdout=subprocess.PIPE)
    
        for line in gitrev_out.stdout:
            hash_=line
            break
        if commit_hash and commit_hash != hash_:
            print("Found multiple configurations with different commit hash")
            sys.exit(1)
        gitrev_out.wait()
        commit_hash = hash_
        print(hash_)
   
    processes=[] 
    for target, prec, std in product(targets, precs, stds):
        print(target, prec, std)
    
    
        cmd='./jenkins_perftest.sh --target '+target+' --std '+std+' --prec '+prec+' --jplan '+jplan+' --outfile out_' +target+'_'+std+'_'+prec+'.log --json '+json_file +' --gtype '+gtype
    
        print("Executing conf : " + target+","+prec+","+std)
        print(cmd)
        processes.append( subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE) )

    for process in processes:
        stdout = process.communicate()
   

    out_jsonfiles=[] 
    for target, prec, std in product(targets, precs, stds):
        outdir=build_outdir(gtype,target,prec,std)
        if not os.path.isdir(outdir):
            print("Output directory: "+outdir+" not found")
            sys.exit(1)
        out_jsonfile=outdir+'/'+json_file_out
        if not os.path.isfile(out_jsonfile):
            print("Output json data: "+ out_jsonfile+" not found")
            sys.exit(1)
        out_jsonfiles.append(out_jsonfile)
    
    print('Merging.................')
    cmd_merge='python merge_updates.py '+json_file+' --updates '+' '.join(out_jsonfiles)
    print(cmd_merge) 
    process = subprocess.Popen(cmd_merge, shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()
    print(stdout)
    shutil.copyfile('stencils.json.merge', json_file)
    print('Finish merging')
 
    f = open(json_file,'r')
    decode = json.load(f)
    
    #Update the hash
    decode['hash'] = commit_hash
    
    f.close()
    fw = open(json_file,'w')
    fw.write(json.dumps(decode,  indent=4, separators=(',', ': ')) )
    fw.close()
    
    for target, prec, std in product(targets, precs, stds):
        print('Copying to PROJECTS...')
        
        outdir=build_outdir(gtype, target,prec,std)
        dst_dir='/project/c01/GridTools/perf_data/'+jplan
        if not os.path.exists(dst_dir+'/'+outdir):
            os.makedirs(dst_dir+'/'+outdir)
        dutil.copy_tree(outdir, dst_dir+'/'+outdir)
     

