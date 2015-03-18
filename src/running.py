# Experiment setup for a single simulation of a day.
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
trace.append({'traceId': 'patient3', 'name': 'Blood Draw','timestamp': 1395441500,'duration': 900, 'resources': ['RNTanya'], 'start_time': 1395411500, 'end_time': 1395411950})

trace.append({'traceId': 'patient3', 'name': 'Exam','timestamp': 1395453200,'duration': 1800, 'resources': ['MDRaymond','MDVictor','MDElaine'], 'start_time': 1395411500, 'end_time': 1395412950})

trace.append({'traceId': 'patient3', 'name': 'Procedure','timestamp': 1395458600,'duration': 7200, 'resources': ['RNMichael'], 'start_time': 1395411500, 'end_time': 1395421950})
traces.append(trace)

log = pmlab.log.EnhancedLog(filename='schedule.xes', cases=traces)

print str(log)

normalized_log = unfolder.normalize_resources(log)

unfold = unfolder.Unfolder()
[net, task_sets, queues] = unfold.unfold(normalized_log)

enricher = unfolder.Enricher(DistributionType.kernel_density_estimate)
net = enricher.enrich(net, normalized_log)

import cProfile
import my_experiment

my_experiment.run_experiment()

