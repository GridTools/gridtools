#!/usr/bin/env python3
import json
import argparse
import os
import copy
import socket
import re

"""
"""
if __name__ == "__main__":

    tolerance = 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument("json_ref", help="filename containing the stencils json report")
    parser.add_argument('--updates',nargs='+', type=str, help='list of update json reports. Each update a particular node')

    args = parser.parse_args()

    f = open(args.json_ref,'r')
    decode = json.load(f)

    f.close()

    host = socket.gethostname()
    if not re.match('greina', host) and not re.match('kesch', host):
        sys.exit('WARNING: host '+host+' not known. Not loading any environment')
    if re.match('greina', host):
        host='greina'
    elif re.match('kesch', host):
        host='kesch'


    if not args.updates:
        parser.error('Need to provide at least one update json report with --updates')

    update_reports = []
    for ufile in args.updates:
        f = open(ufile,'r')
        update_reports.append(json.load(f))
        f.close()

    copy_ref = copy.deepcopy(decode)
    stella_timers = {}

    for ahost in decode:
        if ahost != host: continue
        for stencil in decode[ahost]['stencils'].items():
            for target in stencil[1].items():
                if target[0] != "gpu" and target[0] != "cpu": continue
                for prec in target[1].items():
                    for std in prec[1].items():
                        for thread in std[1].items():
                            for domain in thread[1].items():
                                ref_time = domain[1]["time"]  
                                differ = False
                                cnt=0
                                for update in update_reports:
                                    update_time = update[ahost]['stencils'][stencil[0]][target[0]][prec[0]][std[0]][thread[0]][domain[0]]["time"]
                                    update_rms = update[ahost]['stencils'][stencil[0]][target[0]][prec[0]][std[0]][thread[0]][domain[0]]["rms"]
                                    if update_time != ref_time :
                                        if differ:
                                            print("Error: multiple update reports modify the same token of metrics")
                                            sys.exit(1)
                                        differ = True
                                        print("Found update for :"+stencil[0]+","+target[0]+","+prec[0]+","+std[0]+","+thread[0]+","+domain[0]+ " in report "+args.updates[cnt])
                                        copy_ref[ahost]['stencils'][stencil[0]][target[0]][prec[0]][std[0]][thread[0]][domain[0]]["time"] = update_time
                                        copy_ref[ahost]['stencils'][stencil[0]][target[0]][prec[0]][std[0]][thread[0]][domain[0]]["rms"] = update_rms
                                    cnt=cnt+1

    merged_filename = "stencils.json.merge"
    f = open(merged_filename,"w")
    f.write(json.dumps(copy_ref,  indent=4, separators=(',', ': ')) )
    f.close()

    print("\n\n********************\n********************\nFinal merged report generated in "+merged_filename)
