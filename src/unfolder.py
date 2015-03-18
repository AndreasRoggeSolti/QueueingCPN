# coding: utf-8
# Unfolding (Discovery), Enrichment, and Folding functionality for navigating in the QCSPN model space.
# Methods are described in the paper: "Data-Driven Performance Analysis of Scheduled Processes" submitted to BPM'15.

from numpy.distutils.system_info import lapack_src_info
import os
import pmlab
import pmlab.log
import pmlab.log.reencoders

from datetime import datetime, timedelta

from enum import Enum

from snakes.nets import *
import snakes.plugins
from sortedcontainers import SortedList, SortedSet, SortedDict

snakes.plugins.load("gv", "snakes.nets", "nets")
from nets import *

snakes.plugins.load('time_cpn', 'snakes.nets', 'time_cpn')
from snakes.plugins.time_cpn import Distribution, DistributionType, TimeSimulator, TokenType
from time_cpn import Token, Transition, Place

import collections
import profile
import numpy
import re  # regular expressions


class State(Enum):
    queued = 1
    service = 2
    finished = 3


class Action(Enum):
    enter = 1
    exit = 2


class FoldingOperation(Enum):
    remove_parallelism = 1
    remove_shared_resources = 2
    fuse_queues_and_resources = 3
    remove_individual_information = 4
    remove_resources = 5


class TaskSet(object):
    def __init__(self, net, task_set, structure):
        """Creates a new task set with the queuing construct.

        """
        self.net = net
        self.task_set = task_set
        self.structure = structure


class Unfolder(object):
    """The class that creates a Petri net from a schedule.
    Usage:
    >>> unfolder = Unfolder()
    # create the instance
    >>> unfolder.unfold(log)
    """
    def __init__(self):
        self.fuse_queues = False
        self.net = None
        self.resources_used = []
        self.resource_places = {}  # a dict storing the resources
        self.task_queues = {}  # a dict storing the structure of a task per resource combination
        # (i.e queuing place, entrance transition, service place, exit transition, finished place)
        pass

    def cleanup(self):
        self.net = None
        self.resources_used = []
        self.task_queues = {}
        self.resource_places = {}

    @staticmethod
    def get_overlapping_tasks(trace):
        """Transitively adds running tasks to the running set of tasks to build parallel groups
        of running tasks.
        Example tasks [A,B,C,D1,D2,E]: ( -> time -> )
        [-------A-------]
                 [-----B------]
                        [--C--]
                                [-D1-]    [----D2----]
                                            [-E-]
        will yield three sets of concurrently active tasks:
        [{A,B,C}, {D1}, {D2,E}]
        """
        overlapping_tasks = []
        current_time = 0
        currently_active_tasks = []  # list of concurrently planned tasks
        # end_time = datetime(1970,1,1,12,30)
        end_time = 0  # last time of the currently active concurrent tasks

        # assume tasks are ordered by start time!
        for task in iter(trace):
            task_start_time = task.get('timestamp', 0)
            if type(task_start_time) == str:
                timestr = str(task_start_time[0:19])
                #print timestr
                task_start_time = datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%S')
            duration = task.get('duration', 5)
            if duration > 0:
                task_end_time = task_start_time + timedelta(seconds=duration).total_seconds()
            if end_time > task_start_time:  # belongs to the currently active set
                currently_active_tasks.append(task)
                if task_end_time > end_time:
                    end_time = max(end_time, task_end_time)
            else:  # belongs to a new set (will become the only one currently enabled)
                if currently_active_tasks:
                    overlapping_tasks.append(currently_active_tasks)
                currently_active_tasks = [task]
                end_time = task_end_time
        overlapping_tasks.append(currently_active_tasks)
        return overlapping_tasks

    @staticmethod
    def get_task_place_name(task, state):
        return "p{}_{}".format(task, state.name)

    @staticmethod
    def get_resource_place_name(resource):
        return "p{}".format(resource)

    @staticmethod
    def get_resource_token_name(resource):
        return "r{}".format(resource)

    @staticmethod
    def get_transition_name(task, action):
        return "t{}_{}".format(task, action.name)

    def get_coded(self, task='DUMMY', resources=[]):
        # coded_list = []
        # for i, key in enumerate(self.resources_used):
        #     if key in resources:
        #         coded_list.append(i)
        # return "{}._{}".format(task, ".".join(str(x) for x in coded_list))
        return "{}.{}".format(task, "".join(sorted(resources)))

    def connect_tasks(self, last_task_connectors, new_task_connectors, task_start, trace_name, trace_token, scheduled=True):
        """Establishes a routing connection for a certain trace
        """
        first = not last_task_connectors[0]
        if first:  # the first task gets the token
            # trace_token.set_time(task_start)
            last_task_connectors[1].add(trace_token)

        # add a routing scheduled transition from the last task
        p_last_finished = last_task_connectors[1]
        t_name = '{}_{}_{}'.format(trace_name, p_last_finished.get_name(), new_task_connectors[0].get_name())
        if scheduled:
            self.net.add_transition(Transition(t_name, guard=Expression("pat=='{}'".format(trace_name)),
                                            dist=Distribution(DistributionType.scheduled, mintime=task_start)))
        else:
            self.net.add_transition(Transition(t_name, guard=Expression("pat=='{}'".format(trace_name)),
                                            dist=Distribution(DistributionType.immediate)))
        self.net.add_input(p_last_finished.get_name(), t_name, Variable('pat'))
        self.net.add_output(new_task_connectors[0].get_name(), t_name, Expression('pat'))

        return new_task_connectors

    def unfold(self, schedule_log, fuse_queues=False, scheduled_start=False, scheduled=True, resource_capacity={}):
        """Unfolds the log into a colored Petri net model
        capturing resource dependencies

        returns: - the net,
                 - the list of tasks sets (for all traces),
                 - the task queues assigning each task_name to the net constructs (queuing stations)
        """
        self.cleanup()
        self.fuse_queues = fuse_queues

        self.net = PetriNet(schedule_log.filename)

        task_sets = []
        trace_counter = 0

        arrival_place = Place("arrivals")
        self.net.add_place(arrival_place)
        exit_place = Place("exits")
        self.net.add_place(exit_place)

        for trace in iter(schedule_log.get_cases(full_info=True)):
            trace_counter += 1
            trace_name = trace[0].get('traceId', 'pat{}'.format(trace_counter))
            if scheduled_start:
                trace_token = Token(trace_name, time=trace[0].get('timestamp', 0), schedule=self.get_schedule(trace), type=TokenType.patient, start_end=Unfolder.get_start_end(trace))
            else:
                trace_token = Token(trace_name, time=trace[0].get('start_time', 0), schedule=self.get_schedule(trace), type=TokenType.patient, start_end=Unfolder.get_start_end(trace))

            last_task_connectors = [None, arrival_place]
            concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
            concurrent_set_id = 0
            for task_set in iter(concurrent_task_sets):
                concurrent_set_id += 1
                task = task_set[0]
                task_start = task.get('timestamp', 0)
                if len(task_set) > 1:
                    p_entrance = Place("{}_in_{}".format(trace_name, concurrent_set_id))
                    self.net.add_place(p_entrance)
                    p_exit = Place("{}_out_{}".format(trace_name, concurrent_set_id))
                    self.net.add_place(p_exit)

                    ins_and_outs = []
                    for task in task_set:
                        new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors, trace_counter, concurrent_set_id, resource_capacity=resource_capacity)
                        ins_and_outs.append(new_task_connectors)

                    # create the split transition + scheduling transitions
                    t_name = "split_{}_{}".format(trace_name, concurrent_set_id)
                    self.net.add_transition(Transition(t_name, guard=Expression("pat=='{}'".format(trace_name)), dist=Distribution(DistributionType.immediate)))
                    self.net.add_input(p_entrance.get_name(), t_name, Variable('pat'))
                    for i, in_and_out in enumerate(ins_and_outs):
                        this_task = task_set[i]
                        p_split = Place("{}_splitted_{}_{}".format(trace_name, concurrent_set_id, i))
                        self.net.add_place(p_split)
                        self.net.add_output(p_split.get_name(), t_name, Expression('pat'))
                        self.connect_tasks([p_entrance, p_split], in_and_out,
                                           this_task.get('timestamp', 0), trace_name, trace_token, scheduled=scheduled)
                    # create the join transition to synchronize continuations
                    t_name = "join_{}_{}".format(trace_name, concurrent_set_id)
                    self.net.add_transition(Transition(t_name, guard=Expression("pat=='{}'".format(trace_name)), dist=Distribution(DistributionType.immediate)))
                    for in_and_out in ins_and_outs:
                        try:
                            self.net.add_input(in_and_out[1].get_name(), t_name, Variable('pat'))
                        except ConstraintError:
                            print "is some thing wrong with this resource?"
                    self.net.add_output(p_exit.get_name(), t_name, Expression('pat'))
                    new_task_connectors = tuple([p_entrance, p_exit])
                    last_task_connectors = self.connect_tasks(last_task_connectors, new_task_connectors,
                                                              task_start, trace_name, trace_token, scheduled=scheduled)
                else:
                    new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors, trace_counter, concurrent_set_id, resource_capacity=resource_capacity)
                    last_task_connectors = self.connect_tasks(last_task_connectors, new_task_connectors,
                                                              task_start, trace_name, trace_token, scheduled=scheduled)
                task_sets.append(TaskSet(self.net, task_set, new_task_connectors))
            # after setting up the patient path, we add an immediate transition to the exit place
            t_name = "exit.{}".format(trace_counter)
            self.net.add_transition(Transition(t_name, guard=Expression("pat=='{}'".format(trace_name)), dist=Distribution(DistributionType.immediate)))
            self.net.add_input(last_task_connectors[1].get_name(), t_name, Variable('pat'))
            self.net.add_output(exit_place.get_name(), t_name, Expression('pat'))

        return self.net # , task_sets, self.task_queues

    def add_or_wire(self, task, trace_name, trace_token, last_task_connectors, trace_counter, concurrent_set_id, resource_capacity={}):

        task_duration = task.get('duration', 1)
        resources = task.get('resources', [])

        # gather used resources
        task_resource_places = {}
        for res in resources:
            if res not in self.resource_places:
                place = Place(self.get_resource_place_name(res), [])
                if '$' in res:
                    # remove shared resources:
                    resources_to_join = str(res).split("$")
                    res_capacity = 100000000
                    for resource_to_join in resources_to_join:
                        res_capacity = min(res_capacity, resource_capacity.get(resource_to_join,1))
                else:
                    res_capacity = resource_capacity.get(res,1)
                for r in range(1,res_capacity+1):
                    tok = Token(self.get_resource_token_name(res), time=0.0)
                    place.add(tok)
                self.net.add_place(place)
                self.resource_places[res] = place
                self.resources_used.append(res)
            task_resource_places[res] = self.resource_places.get(res)

        if self.fuse_queues:
            key = self.get_coded(task.get('name', 'DUMMY'), resources)
        else:
            task_name = "{}.{}.{}".format(self.get_coded(task.get('name', 'DUMMY'), resources), trace_counter, concurrent_set_id)
            key = task_name

        # add task to net
        if key not in self.task_queues:
            # create places for task:
            p_names = {State.queued: self.get_task_place_name(key, State.queued),
                       State.service: self.get_task_place_name(key, State.service),
                       State.finished: self.get_task_place_name(key, State.finished)}
            p_queue = Place(p_names[State.queued], [])
            self.net.add_place(p_queue)
            p_service = Place(p_names[State.service], [])
            self.net.add_place(p_service)
            p_finish = Place(p_names[State.finished], [])
            self.net.add_place(p_finish)
            # create transitions for task:
            # enter:
            t_names = {Action.enter: self.get_transition_name(key, Action.enter),
                       Action.exit: self.get_transition_name(key, Action.exit)}
            self.net.add_transition(Transition(t_names[Action.enter], guard=Expression("True"), dist=Distribution(DistributionType.deterministic, time=task.get('duration',1))))
            self.net.add_input(p_names[State.queued], t_names[Action.enter], Variable('pat'))
            tuple_parts = [Expression('pat')]
            variables = [Variable('pat')]
            counter = 0
            for res, task_res in task_resource_places.iteritems():
                counter += 1
                res_name = 'res{}'.format(counter)
                self.net.add_input(self.get_resource_place_name(res), t_names[Action.enter], Variable(res_name))
                tuple_parts.append(Expression(res_name))
                variables.append(Variable(res_name))
            self.net.add_output(p_names[State.service], t_names[Action.enter], Tuple(tuple_parts))
            # exit:
            self.net.add_transition(Transition(t_names[Action.exit], guard=Expression("True"), dist=Distribution(DistributionType.immediate)))
            self.net.add_input(p_names[State.service], t_names[Action.exit], Tuple(variables))
            self.net.add_output(p_names[State.finished], t_names[Action.exit], Expression('pat'))
            counter = 0
            for res, task_res in task_resource_places.iteritems():
                counter += 1
                res_name = 'res{}'.format(counter)
                self.net.add_output(self.get_resource_place_name(res), t_names[Action.exit], Variable(res_name))
            self.task_queues[key] = {State.queued: p_queue,
                                State.service: p_service,
                                State.finished: p_finish,
                                Action.enter: t_names[Action.enter],
                                Action.exit: t_names[Action.exit]}
        else: # task is already added to the net.
            pass
        p_queue = self.task_queues[key][State.queued]
        p_finish = self.task_queues[key][State.finished]
        return tuple([p_queue, p_finish])

    def get_schedule(self, trace):
        schedule = []
        for task in trace:
            key = self.get_coded(task.get('name', 'DUMMY'), task.get('resources', []))
            schedule.append(key)
        return schedule

    @staticmethod
    def get_start_end(trace):
        start = trace[0].get('start_time', 0)
        end = trace[-1].get('end_time', 0)
        return [start,end]

class Folder(object):
    def __init__(self):
        self.net = None
        self.task_sets = None
        self.task_queues = None

    def cleanup(self):
        self.net = None
        self.task_sets = None
        self.task_queues = None

    @staticmethod
    def fold_log(log, operation):
        new_traces = []
        if operation == FoldingOperation.remove_parallelism:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
                for task_set in iter(concurrent_task_sets):
                    if len(task_set) > 1:
                        task_merged_name = ""  # build task name
                        resources = set([])
                        duration = 0
                        times = []
                        start_time = sys.maxint
                        end_time = 0
                        for task in task_set:
                            task_merged_name = task_merged_name + task.get('name', 'DUMMY')
                            task_resources = task.get('resources', [])
                            task_duration = task.get('duration', 1)
                            resources = resources.union(set(task_resources))
                            duration = max(duration, task_duration)
                            times.append(task.get('timestamp',0))
                            start_time = min(start_time, task.get('start_time'))
                            end_time = max(end_time, task.get('end_time'))
                        timestamp = min(times)
                        new_trace.append({'name': task_merged_name, 'timestamp': timestamp, 'duration': duration, 'resources': resources, 'start_time': start_time, 'end_time': end_time})
                    else:
                        new_trace.append(task_set[0])
                new_traces.append(new_trace)
        elif operation == FoldingOperation.remove_shared_resources:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
                for task_set in iter(concurrent_task_sets):
                    for task in task_set:
                        task_resources = task.get('resources', [])
                        if len(task_resources) > 1:
                            # merge them
                            new_resource = "$".join(sorted(task_resources))
                            task['resources'] = [new_resource]
                        new_trace.append(task)
                new_traces.append(new_trace)
        elif operation == FoldingOperation.remove_resources:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                for task in trace:
                    new_trace.append({'name': task.get('name', 'DUMMY'), 'timestamp': task.get('timestamp',0), 'duration': task.get('duration',1), 'resources': [], 'start_time': task.get('start_time'), 'end_time': task.get('end_time')})
                new_traces.append(new_trace)
        return pmlab.log.EnhancedLog(filename=log.filename, cases=new_traces)


    def fold(self, net, task_sets, task_queues, operation):
        """Folds a net according to an operation FoldingOperation
         task_sets contains All task sets in PI of all traces
         task_queues contains the corresponding model structures
         Idea is to return again the triple of the (folded) net, the remaining task_sets, and the remaining task_queues
        """
        self.net = net
        self.task_sets = task_sets
        self.task_queues = task_queues
        if operation == FoldingOperation.remove_parallelism:
            self.remove_parallelism()
        elif operation == FoldingOperation.remove_shared_resources:
            self.remove_shared_resources()
        elif operation == FoldingOperation.fuse_queues_and_resources:
            self.merge_resources()
            self.fuse_queues()
        elif operation == FoldingOperation.remove_individual_information:
            self.remove_individual_information()
        return self.net, self.task_sets, self.task_queues

    def remove_parallelism(self):
        """Removes concurrency by joining all parallel tasks into a single big task.
         (Time is computed as the maximum of individual tasks)
        """
        pass
        # for i, task_set in enumerate(self.task_sets):
        #     if len(task_set) > 1:
        #         task_merged_name = ""  # build task name
        #         for task in task_set:
        #             task_merged_name = task_merged_name + task.get('name', 'DUMMY')
        #         self.net.add_transition(Transition(t_names[Action.enter], guard=Expression("True"), dist=Distribution(DistributionType.exponential, rate=1)))
        #         # merged_task =
        #         new_task_set = {}
        #         self.task_sets[i] = self.new_task_set
        #     else:
        #         pass  # nothing to do for single tasks

    def remove_shared_resources(self):
        """Removes shared resources by joining them to one (copied) resource
        """
        # todo implement
        pass

    def merge_queues(self):
        """Merges queues of the same task/resource
        adds routing transitions after merging the queues
        """
        # todo implement
        pass

    def merge_resources(self):
        """Merges resources of the same kind -> here "kind" is defined by the role these resources play
         that is, all resources performing "blood draw" will be collected in one single place with multiple tokens
        """
        # todo implement
        pass

    def remove_individual_information(self):
        """Fuses queues of the same task/resource
        adds routing transitions after merging the queues
        """
        # todo implement
        pass


class Enricher(object):
    def __init__(self, distribution_type):
        self._dist_type = distribution_type

    def enrich(self, cpn, log):
        # for all task/resource combinations
        durations = {}  # stores real durations per key (task x resource²)
        scheduled_durations = {} # stores scheduled durations per key (task x resource²)
        for trace in log:
            for event in trace:
                #key = "{}.{}".format(event['name'], "".join(sorted(event['resources'])))
                key = event['name']
                value = event.get('end_time', 0) - event.get('start_time', 0)
                if key not in durations.keys():
                    durations[key] = [value]
                    scheduled_durations[key] = [event.get('duration')]
                else:
                    durations[key].append(value)
                    scheduled_durations[key].append(event.get('duration'))

        for transition in cpn.transition():
            t_name = str(transition)
            if t_name.endswith("_enter"):
                # enrich transition with historical times
                key = ".".join(t_name[1:-6].split(".")[:1])
                if key not in durations.keys():
                    values = [0]
                    pass
                else:
                    values = durations[key]
                dist = None
                if len(values) < 2  or self.same_values(values):
                    if key in scheduled_durations:
                        dist = Distribution(dist_type=DistributionType.exponential, rate=1.0/scheduled_durations[key][0])
                    # if values[0] > 0:
                    #     dist = Distribution(dist_type=DistributionType.exponential, rate=1./values[0])
                    # else:
                    #     dist = Distribution(dist_type=DistributionType.exponential, rate=1)
                else:
                    dist = Distribution(dist_type=DistributionType.empirical, fit=True, values=values)
                if dist is not None:
                    transition.set_dist(dist)
        return cpn

    def same_values(self, values):
        if len(values) == 0:
            return True
        value = values[0]
        for v in values:
            if v != value:
                return False
        return True



def normalize_resources(log):
    """Replaces duplicate log events that use different resources with one entry that uses the set of resources
    assumption: resources are stored in the "resource" key
    """
    # go through all cases
    resource_capacity = collections.defaultdict(int) # resource_capacity['RNTanya'] = 5
    all_events = []

    for case in log:
        last_event = None
        for event in case:
            all_events.append({'timestamp':event.get('timestamp'), 'resource':event.get('resource'), 'in':True})
            all_events.append({'timestamp':(int(event.get('timestamp'))+int(event.get('duration'))), 'resource':event.get('resource'), 'in':False})
            if 'resource' in event:
                event['resources'] = []
                if last_event and last_event['name'] == event['name']:
                    # TODO: make sure that also times match!
                    last_event['delete'] = True
                    event['resources'] = list(last_event['resources'])
                if event['resource']:
                    event['resources'].append(str(event['resource']))
                del event['resource']
                last_event = event
        case[:] = [evt for evt in case if 'delete' not in evt]

    all_events_sorted = sorted(all_events, key=lambda k: k['timestamp'])
    current_utilization = collections.defaultdict(int)
    for ev in all_events_sorted:
        if isinstance(ev.get('resource'), list):
            # empty
            pass
        else:
            if ev.get('in'):
                current_utilization[ev.get('resource')] += 1
                resource_capacity[ev.get('resource')] = max(current_utilization[ev.get('resource')], resource_capacity[ev.get('resource')])
            else:
                current_utilization[ev.get('resource')] -= 1

    return log, resource_capacity


def extract_task_name(task_name):
    """Returns the name of a task without the id that is appended.
    Example: given "BloodDraw.72" it returns "BloodDraw"
    Used in fusing tasks that have the same name
    """
    trimmed = re.sub('\.[0-9]+$', '', task_name)
    trimmed
