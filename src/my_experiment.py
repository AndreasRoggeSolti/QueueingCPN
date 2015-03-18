# Experiment setup for a repeated simulation of a single day.

import sys
import os
import pmlab
import pmlab.log
import pmlab.log.reencoders

from datetime import datetime, timedelta

from enum import Enum

from snakes.nets import *
import snakes.plugins

snakes.plugins.load('time_cpn', 'snakes.nets', 'time_cpn')
from snakes.plugins.time_cpn import Distribution, DistributionType, TimeSimulator

import unfolder
import time

import cProfile


def perform_run(models, run, f):
    f.write("model_name; patient; act_start; act_end; pred_start; pred_end; run_num\n")
    for name, model in models.iteritems():
        m = model.get_marking()
        initial_marking = m.copy()
        sim = TimeSimulator(net=model)
        if name == "QCSPN rp,rsr,fq":
            sim.snapshot_principle = True
        marking = sim.simulate_one(marking=initial_marking.copy())

        for m_set in marking.values():
            for token in m_set.items():
                if snakes.plugins.time_cpn.is_patient(token):
                    # print "{}\n{}\n".format(token, token.export_history())
                    f.write(token.export_start_end(name, run=run))
                    f.write("\n")
                    if sim.snapshot_principle:
                        f.write(token.export_snapshot_predictor("QCSPN rp,rsr,fq,snapshot", run=run))
                        f.write("\n")
        # restore initial marking for next iterations
        model.set_marking(initial_marking)
        f.flush()


def run_experiment(num_runs=30):

    traces = []
    # Patient 1 - Blood Draw, Exam, Procedure (sequence)
    trace=[]
    trace.append({'traceId': 'patient1', 'name': 'Blood Draw','timestamp': 1393411500,'duration': 900, 'resources': ['RNTanya'], 'start_time': 1393411500, 'end_time': 1393411900})

    trace.append({'traceId': 'patient1', 'name': 'Exam','timestamp': 1393423200,'duration': 1800, 'resources': ['MDRaymond','MDVictor','MDElaine'], 'start_time': 1393411500, 'end_time': 1393412900})

    trace.append({'traceId': 'patient1', 'name': 'Procedure','timestamp': 1393428600,'duration': 7200, 'resources': ['RNMichael'], 'start_time': 1393411500, 'end_time': 1393421900})
    traces.append(trace)

    # Patient 2 - Blood Draw, Exam, Speech Therapy (in parallel)
    trace=[]
    trace.append({'traceId': 'patient2', 'name': 'Blood Draw','timestamp': 1393413000,'duration': 600, 'resources': ['RNTanya'], 'start_time': 1393410500, 'end_time': 1393411950})

    trace.append({'traceId': 'patient2', 'name': 'Exam','timestamp': 1393419600,'duration': 1800, 'resources': ['MDElaine'], 'start_time': 1393411500, 'end_time': 1393412350})

    trace.append({'traceId': 'patient2', 'name': 'Speech Therapy','timestamp': 1393419600,'duration': 1800, 'resources': ['STBrooke'], 'start_time': 1393421500, 'end_time': 1393422650})
    traces.append(trace)

    # Patient 3 - same as Patient 1
    trace=[]
    trace.append({'traceId': 'patient3', 'name': 'Blood Draw','timestamp': 1395441500,'duration': 900, 'resources': [], 'start_time': 1395411500, 'end_time': 1395411950})

    trace.append({'traceId': 'patient3', 'name': 'Exam','timestamp': 1395453200,'duration': 1800, 'resources': ['MDRaymond','MDVictor','MDElaine'], 'start_time': 1395411500, 'end_time': 1395412950})

    trace.append({'traceId': 'patient3', 'name': 'Procedure','timestamp': 1395458600,'duration': 7200, 'resources': ['RNMichael'], 'start_time': 1395411500, 'end_time': 1395421950})
    traces.append(trace)

    log = pmlab.log.EnhancedLog(filename='schedule.xes', cases=traces)

    print str(log)

    start_time = time.time()

    normalized_log = unfolder.normalize_resources(log)

    unfold = unfolder.Unfolder()
    [net, task_sets, queues] = unfold.unfold(normalized_log)

    enricher = unfolder.Enricher(DistributionType.kernel_density_estimate)
    net = enricher.enrich(net, normalized_log)

    # folder = unfolder.Folder()

    log_rp = unfolder.Folder.fold_log(normalized_log, unfolder.FoldingOperation.remove_parallelism)
    [net_rp, task_sets_rp, queues_rp] = unfold.unfold(log_rp)
    net_rp = enricher.enrich(net_rp, log_rp)


    log_rp_rsr = unfolder.Folder.fold_log(log_rp, unfolder.FoldingOperation.remove_shared_resources)
    [net_rp_rsr, task_sets_rp_rsr, queues_rp_rsr] = unfold.unfold(log_rp_rsr)
    net_rp_rsr = enricher.enrich(net_rp_rsr, log_rp_rsr)

    [net_rp_rsr_fq, task_sets_rp_rsr_fq, queues_rp_rsr_fq] = unfold.unfold(log_rp_rsr, fuse_queues=True)
    net_rp_rsr_fq = enricher.enrich(net_rp_rsr_fq, log_rp_rsr)

    net.draw("model_running_init.png")
    net_rp.draw("model_running_rp_init.png")
    net_rp_rsr.draw("model_running_rp_rsr_init.png")
    net_rp_rsr_fq.draw("model_running_rp_rsr_fq_init.png")

    models = {"QCSPN orig": net, "QCSPN rp": net_rp, "QCSPN rp,rsr": net_rp_rsr, "QCSPN rp,rsr,fq": net_rp_rsr_fq}



    print "Finished model construction."
    print("--- %s seconds ---" % (time.time() - start_time))

    f = open('output.csv', 'w')

    before_simulation = time.time()
    for run in range(num_runs):
        start_time = time.time()
        perform_run(models, run, f)
        print("---- run {} completed in %s seconds ----".format(run) % (time.time() - start_time))

    print("---- all runs completed in %s seconds ----".format(run) % (time.time() - before_simulation))

    # show final state:
    sim = TimeSimulator(net=net)
    marking = sim.simulate_one()
    net.draw("model_running_end.png")


