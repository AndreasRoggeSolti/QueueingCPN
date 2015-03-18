"""A plugin that enriches tokens with timing information."""

import snakes.plugins
import numpy as np
import re
from scipy import stats
import math
from scipy.stats import norm, expon

from snakes.lang import *
from snakes.data import *

from sortedcontainers import SortedList, SortedDict, SortedSet
from collections import defaultdict

import random
import time as timer
import datetime

from enum import Enum

# @snakes.plugins.plugin("snakes.data")
# def extend(module):
# "Extends `module`"


#class MultiSet(snakes.data.MultiSet):
#    def add_time(self, time):
#        for tok in self.items():
#            tok.add_time(time)
#    return MultiSet

@snakes.plugins.plugin("snakes.nets")
def extend(module):
    """Extends `module`"""

    class Token(module.Token):
        """Extension of the class `Token` in `module`"""

        def __init__(self, value, **kwargs):
            """Add new time parameter `time`

            >>> print Token(1).get_time()
            0
            >>> print Token(1, time=1000).get_time()
            1000

            @param kwargs: plugin options
            @keyword time: the time of the token.
            @type hello: `float`
            """
            self._time = kwargs.pop("time", 0)
            self._history = []  # the firing history
            self._schedule = kwargs.pop("schedule", [])
            self.start_end = kwargs.pop("start_end", [0,0])
            self.type = kwargs.pop("type", TokenType.resource)
            self._snapshot_estimate = 0
            self._queueing_estimate = 0
            self._mixed_estimate = 0
            module.Token.__init__(self, value)

        def get_time(self):
            return self._time

        def set_time(self, time):
            self._time = time

        def add_time(self, time):
            self._time += time

        def trace_firing(self, firing):
            """
            Records a specific firing in the token's history
            expects something like:
            >>> token = Token(42)
            >>> token.trace_firing(FiringEvent('A', 13588182, 500})
            :param firing: a FiringEvent containing information of the firing
            """
            self._history.append(firing)

        def export_duration(self, model_name="Default"):
            """
            returns a string representing the total duration in the model:
            time of last transition, which is the exit - time of first transition

            :param model_name:
            :return: duration in seconds
            """
            duration = 0 if len(self._history) < 2 else self._history[-1].get_time() - self._history[0].get_time()
            return "{};{};{}".format(model_name, self.value, duration)

        def export_start_end(self, model_name="Default", **kwargs):
            """
            returns a string representing the total duration in the model:
            time of last transition, which is the exit - time of first transition

            :param model_name:
            :return: duration in seconds
            """
            pred_start_end = [0, 0] if len(self._history) < 2 else [self._history[0].get_time(), self._history[-1].get_time()]
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(pred_start_end[0]), str(pred_start_end[1])] + kwargs.values()
            return ";".join(map(str, vals))

        def export_snapshot_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self._snapshot_estimate < 0 else start + self._snapshot_estimate
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(start), str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_queueing_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self._queueing_estimate < 0 else start + self._queueing_estimate
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(start), str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_mixed_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self.get_mixed_estimate() < 0 else start + self.get_mixed_estimate()
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(start), str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_history(self):
            return "\n".join(str(hist) for hist in self._history)

        def get_last_sojourn_time(self):
            if len(self._history) < 3:
                raise Exception("token {} doesn't have enough firing history to compute sojourn time!".format(str(self)))
            return self._history[-1].get_time() - self._history[-3].get_time()

        def get_snapshot_estimate(self):
            return self._snapshot_estimate

        def get_queueing_estimate(self):
            return self._queueing_estimate

        def get_mixed_estimate(self):
            if self.get_snapshot_estimate() <0:
                return self.get_queueing_estimate()
            else:
                return self._snapshot_estimate


        def set_snapshot_estimate(self, estimate):
            self._snapshot_estimate = estimate

        def set_queueing_estimate(self, estimate):
            self._queueing_estimate = estimate

        def get_schedule(self):
            return self._schedule

        def __str__(self):
            """Simple string representation (that of the value)

            >>> str(Token(42))
            '42'

            @return: simple string representation
            @rtype: `str`
            """
            return str("{}@{}".format(self.value, self._time))

        def __repr__(self):
            return self.__str__()

        def __eq__(self, other):
            """Tests Tokens for equality.

            >>> Token('joe', time=2) == 'joe'
            True

            @param other: the value to compare against
            @type other: `any`
            @rtype: `Boolean`
            """
            if type(other) is str:
                return self.value == other
            else:
                return self.__cmp__(other) == 0

        def __cmp__(self, other):
            """Compares two tokens by their time value
            :param other:
            :return:
            """
            if hasattr(other, '_time'):
                if self._time == other.get_time():
                    if self.value != other.value:
                        print "stop for a sec."
                    return cmp(self.value, other.value)
                return int(self._time - other.get_time())
            else:
                return -1

    class Place(module.Place):

        def __init__(self, name, tokens=[], check=None):
            self._last_sojourn_time = -1
            module.Place.__init__(self, name, tokens, check)

        def get_tokens(self, global_time):
            tokens = MultiSet([])
            for tok in iter(self.tokens):
                if isinstance(tok, tuple):
                    if tok[0].get_time() <= global_time:
                        tokens.add([tok], 1)
                elif tok.get_time() <= global_time:
                    tokens.add(tok, 1)
            return tokens

        def get_name(self):
            return self.name

        def get_last_sojourn_time(self):
            if self._last_sojourn_time < 0:
                raise ValueError("no last sojourn time recorded")
            else:
                return self._last_sojourn_time

        def set_last_sojourn_time(self, sojourn_time):
            self._last_sojourn_time = sojourn_time

        def reset_last_sojourn_time(self):
            self._last_sojourn_time = -1

    class Transition(module.Transition):
        "Extension of the class `Transition` in `module`"

        expression = re.compile(r'_arrivals_')

        def __init__(self, name, guard=None, **kwargs):
            """Add new time parameter `weight`
            >>> print Transition(1).get_weight()
            1
            >>> print Transition(1, weight=2).get_weight()
            2

            :param kwargs: plugin options
            :keyword weight: the weight of the transition.
            :keyword dist: the distribution of the transition.
            :type hello: `float`
            :type dist: `Distribution`
            """
            # print "transition kwargs {}".format(kwargs)
            self._weight = kwargs.pop("weight", 1)
            self._dist = kwargs.pop("dist", Distribution(DistributionType.exponential))
            module.Transition.__init__(self, name, guard)
            # print "created transition {} with weight: {} and dist: {}".format(name, self._weight, self._dist)

        # def __str__(self):
        #     """Simple string representation (that of the value)
        #
        #     >>> str(Transition('t1'))
        #     't1'
        #
        #     @return: simple string representation
        #     @rtype: `str`
        #     """
        #     return str("({} w:{})".format(self.name, self._weight))

        @staticmethod
        def is_first_transition(name):
            return Transition.expression.search(name) is not None

        def set_dist(self, dist):
            """Sets the distribution for the transition.
            :param dist: the distribution to set
            :type dist: `Distribution`
            """

            self._dist = dist

        def get_dist(self):
            return self._dist

        def get_weight(self):
            return self._weight

        def modes(self, **kwargs):
            """Return the list of bindings which enable the transition.
            Note that the modes are usually considered to be the list of
            bindings that _activate_ a transitions. However, this list may
            be infinite so we restricted ourselves to _actual modes_,
            taking into account only the tokens actually present in the
            input places.
            >>> t = Transition('t', Expression('x!=y'))
            >>> px = Place('px', [Token(0,time=1),Token(1,time=10)])
            >>> t.add_input(px, Variable('x'))
            >>> py = Place('py', [Token(0, time=3), Token(1,time=4)])
            >>> t.add_input(py, Variable('y'))
            >>> m = t.modes(time=5)
            >>> len(m)
            1
            >>> Substitution(y=0, x=1) in m
            True
            >>> Substitution(y=1, x=0) in m
            False

            @return: a list of substitutions usable to fire the transition
            @rtype: `list`
            """
            global_time = kwargs.pop('time', 0)
            # print "global time at binding: {}".format(global_time)
            parts = []
            try:
                for place, label in self.input():
                    m = label.modes(place.get_tokens(global_time))
                    parts.append(m)
            except module.ModeError:
                return []
            result = []
            for x in cross(parts):
                try:
                    if len(x) == 0:
                        sub = Substitution()
                    else:
                        sub = reduce(Substitution.__add__, x)
                    if self._check(sub, False, False):
                        result.append(sub)
                except DomainError:
                    pass
            return result

        def fire(self, binding, **kwargs):
            """Fires a transition and returns the time of availability of the token
            for further progress in the model.
            :return: time
            :rtype: numeric (double)
            """
            if self.enabled(binding):
                sample_time = timer.time()
                global_time = kwargs.get('time')
                duration = int(round(self._dist.sample(time=global_time)))
                # print "- sampling took %s seconds of {}".format(self._dist) % (timer.time() - sample_time)
                if duration > 10*3600:
                    print "loong duration %s hours for {}".format(self.name) % (duration/3600)

                snapshot_principle = kwargs.get('snapshot', False)
                time = global_time + duration
                for place, label in self.input():
                    place.remove(label.flow(binding))
                    for item in label.flow(binding):
                        # print "firing {} {}s: removing {} from\n{}".format(self, duration, item, kwargs.get('tokens', {}))
                        if isinstance(item, tuple):
                            for tok in item:
                                if is_patient(tok):
                                    if len(kwargs.get('tokens', {}).get(tok)) == 1:
                                        kwargs.get('tokens', {}).pop(tok)
                                    else:
                                        kwargs.get('tokens', {}).get(tok).remove(place)
                                    # kwargs.get('tokens', {}).get(tok).remove(place)
                        else:
                            if is_patient(item):
                                if len(kwargs.get('tokens', {}).get(item)) == 1:
                                    kwargs.get('tokens', {}).pop(item)
                                else:
                                    kwargs.get('tokens', {}).get(item).remove(place)
                                # kwargs.get('tokens', {}).get(item).remove(place)

                for place, label in self.output():
                    tokens = label.flow(binding)
                    for tok in tokens:
                        if isinstance(tok, tuple):
                            for t in tok:
                                t.set_time(time)
                                t.trace_firing(FiringEvent(self.name, global_time, duration))
                                if is_patient(t):
                                    # if t not in kwargs.get('tokens', {}):
                                    #     kwargs.get('tokens', {})[t] = []
                                    kwargs.get('tokens', {})[t].append(place)
                        else:
                            tok.set_time(time)
                            tok.trace_firing(FiringEvent(self.name, global_time, duration))
                            if is_patient(tok):
                                # if tok not in kwargs.get('tokens', {}):
                                #     kwargs.get('tokens', {})[tok] = []
                                kwargs.get('tokens', {})[tok].append(place)
                    place.add(tokens)

                    if snapshot_principle:
                        net = kwargs.pop("net", None)
                        patient_token = tokens.items()[0]
                        if place.get_name().endswith("_finished"):
                            # store sojourn time:
                            if len(tokens) > 1:
                                print "debug me!"
                            sojourn_time = patient_token.get_last_sojourn_time()
                            place.set_last_sojourn_time(sojourn_time)
                            # print "place {} has sojourn time {} - {}".format(str(place), sojourn_time, patient_token)
                        elif Transition.is_first_transition(self.name):
                            # print "estimating snapshot for {}".format(patient_token)
                            try:
                                duration = 0
                                for task_name in patient_token.get_schedule():
                                    task_finished_place = net.place("p{}_finished".format(task_name))
                                    duration += task_finished_place.get_last_sojourn_time()
                            except ValueError:
                                duration = -1
                            patient_token.set_snapshot_estimate(duration)

                            queue_duration = 0
                            for task_name in patient_token.get_schedule():
                                task_queued_place = net.place("p{}_queued".format(task_name))
                                task_service_place = net.place("p{}_service".format(task_name))
                                queue_length = 1 + len(task_queued_place.tokens)
                                num_in_service = len(task_service_place.tokens)
                                #in_service = num_in_service==1
                                dist_mean = net.transition("t{}_enter".format(task_name)).get_dist().get_mean()
                                #[task_dec]= task_name.split(".")
                                res_name="p"+task_name.split(".")[1]
                                res_place = net.place(res_name)
                                res_num_idle = len(res_place.tokens)

                                if res_num_idle>0:
                                    queue_duration +=  dist_mean
                                else:
                                    if num_in_service==0:
                                        queue_duration +=  dist_mean
                                    else:
                                        queue_duration +=  dist_mean*queue_length/num_in_service + dist_mean


                                #if in_service:
                                    #dist_cv_sq = math.pow(net.transition("t{}_enter".format(task_name)).get_dist().get_CV(),2)
                                  #  queue_duration +=  dist_mean*queue_length/res_num_ + dist_mean #*(1+dist_cv_sq)/2
                                #else:
                                   # queue_duration +=  dist_mean
                            patient_token.set_queueing_estimate(queue_duration)
                return time
            else:
                raise ValueError("transition not enabled for %s" % binding)

    return Place, Token, Transition


class FiringEvent(object):
    """
    Information collected at tokens
    TODO: We need to merge information on joining parallel tokens!! (if we need more detailed analysis)
    """
    def __init__(self, transition, firing_time, duration):
        self._transition = transition
        self._firing_time = firing_time
        self._duration = duration

    def __str__(self):
        return "Firing; {}; {}; {}".format(self._transition, self._firing_time, self._duration)

    @staticmethod
    def get_header():
        return "Type; Transition; Time; Duration"

    def get_time(self):
        return self._firing_time


class QueueingPolicy(Enum):
    random = 1
    earliest_due_date = 2


class DistributionType(Enum):
    normal = 1
    exponential = 2
    uniform = 3
    scheduled = 4
    immediate = 5
    kernel_density_estimate = 6
    deterministic = 7
    empirical = 8


class TokenType(Enum):
    patient = 0
    resource = 1

class Distribution:
    """
    a class to capture supported distributions that can be fit to data
    """

    def __init__(self, dist_type=DistributionType.exponential, **kwargs):
        if isinstance(dist_type, basestring):
            raise ValueError("Don't use strings, but DistributionTypes to initialize distributions!")
        self._dist = None
        fit = kwargs.get('fit', False)
        if not fit:
            if dist_type == DistributionType.normal:
                self._param = [kwargs.get('mean', 5), kwargs.get('sd', 1)]
            elif dist_type == DistributionType.exponential:
                self._param = [kwargs.get('rate', 1)]
            elif dist_type == DistributionType.uniform:
                self._param = [kwargs.get('low', 0), kwargs.get('high', 1)]
            elif dist_type == DistributionType.scheduled:
                self._param = [kwargs.get('mintime', 0)]
            elif dist_type == DistributionType.deterministic:
                self._param = [kwargs.get('time', 0)]
            else:  # immediate transition
                self._param = [0]
                dist_type = DistributionType.immediate
        else:  # fit distributions to data
            # Assuming that the values are stored in a 1-d vector called "values"
            self._param = kwargs.get('values', [0,0,1])
            if dist_type == DistributionType.empirical:
                pass
            else:
                self._dist = stats.gaussian_kde(self._param)
                dist_type = DistributionType.kernel_density_estimate

        self._name = dist_type
        # print "creating {} distribution with {} values".format(dist_type, self._param)

    def __repr__(self):
        return "{} distribution (params:{})".format(self._name, self._param)

    def sample(self, **kwargs):
        """
        Samples a random value from the distribution that is used.
        :return: a sample from the distribution
        """
        if TimeSimulator.DEBUG:
            print "dist name: {}".format(self._name)
        if self._name == DistributionType.normal:
            return random.gauss(self._param[0], self._param[1])
        elif self._name == DistributionType.exponential:
            return random.expovariate(self._param[0])
        elif self._name == DistributionType.immediate:
            return 0
        elif self._name == DistributionType.scheduled:
            current_time = kwargs.get('time', 0)
            return max(self._param[0], current_time) - current_time
        elif self._name == DistributionType.kernel_density_estimate:
            return abs(self._dist.resample(size=1)[0][0])
        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.empirical:
            # draw one of the past samples uniformly
            return self._param[random.randint(0, len(self._param)-1)]

        else:  # assume uniform distribution
            return random.uniform(self._param[0], self._param[1])

    def get_mean(self):
        if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
            return np.mean(self._param)
        elif self._name == DistributionType.exponential:
            return 1/self._param[0]
        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.immediate:
            return 0
        else:
            print "please implement for the Mean {} - debug me!".format(self._name)
            return -1

    def get_CV(self):
        if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
            if np.mean(self._param)==0:
                print "zero mean for the CV calculation - debug me!"
            else:
                return np.std(self._param)/np.mean(self._param)
        elif self._name == DistributionType.exponential:
             return 1
        elif self._name == DistributionType.deterministic:
            return 0
        elif self._name == DistributionType.immediate:
            return 0
        else:
            print "please implement for the CV {} - debug me!".format(self._name)
            return -1
class TimeSimulator(object):
    MAX_EVENTS_PER_RUN = 10000000

    DEBUG = False

    def __init__(self, net):
        self.net = net
        self.initial = net.get_marking().deepcopy()
        self.current = self.initial
        self.current_time = 0
        self.snapshot_principle = False
        self.task_snapshot = {}
        self.sorted_tokens = defaultdict(list)  # contains a sorted list of tokens and their associated place
        self.place_x_pat2transition = SortedDict()

    def set_snapshot(self, snapshot):
        self.snapshot_principle = snapshot

    def simulate_one(self, marking=None, global_time=0, queuing_policy=QueueingPolicy.earliest_due_date):
        """
        Simulates a schedule-driven queueing-enabled net until all scheduled tasks are done
        (we mostly use it to simulate one day at once)

        :param marking: initial marking
        :param global_time: the global time at start (a low value should be fine)
        :param queuing_policy: the queueing policy (random or EDD)
        :rtype : Marking
        :return: the marking after no more transition is available
        """
        if marking is None:
            marking = self.initial
        self.current = marking.deepcopy()
        self.current_time = global_time

        # cleanup sojourn times of previous runs
        for place in self.net.place():
            place.reset_last_sojourn_time()

        times_to_check = SortedSet(load=100)

        times_to_check.add(global_time)

        for pName, multi_set in self.current.iteritems():
            for tok in iter(multi_set):
                if is_patient(tok):
                    times_to_check.add(tok.get_time())
        if TimeSimulator.DEBUG:
            print "times to check: {}".format(times_to_check)

        self.net.set_marking(self.current)
        if TimeSimulator.DEBUG:
            print "parsing net structure"
            last_time = timer.time()

        self.sorted_tokens = self.collect_tokens_from_net()
        self.parse_net_structure()
        if TimeSimulator.DEBUG:
            print "-- done parsing in %s seconds ---" % (timer.time() - last_time)



        num = 0
        while num < TimeSimulator.MAX_EVENTS_PER_RUN:
            num += 1
            enabled_transitions = {}
            sum_weight = 0

            # look only at tokens that have a time less or equal to the global time:
            available_tokens = self.get_available_tokens_and_places(global_time)
            if len(available_tokens) == 0:
                # global time is below all token's values
                global_time = times_to_check.pop(0)
                print datetime.datetime.fromtimestamp(global_time).strftime('%Y-%m-%d %H:%M:%S')
            else:
                # some tokens are available -> does not mean that transitions can fire as well
                # let's check...
                transitions_to_check = SortedSet([])
                for tok, place in available_tokens.iteritems():
                    for p in place:
                        if 'all' in self.place_x_pat2transition[p.name]:
                            for t_to_check in self.place_x_pat2transition[p.name]['all']:
                                transitions_to_check.add(t_to_check)
                        if tok.value in self.place_x_pat2transition[p.name]:
                            for t_to_check in self.place_x_pat2transition[p.name][tok.value]:
                                transitions_to_check.add(t_to_check)

                for trans in transitions_to_check:
                    modes = trans.modes(time=global_time)
                    if modes:
                        enabled_transitions[trans] = modes
                        sum_weight += trans.get_weight()

                if not enabled_transitions:
                    # increment time to next token's time!
                    last_time = global_time
                    if not times_to_check:
                        break
                    global_time = times_to_check.pop(0)
                    if TimeSimulator.DEBUG:
                        print "no transitions enabled at time {} checking model at time {}".format(last_time, global_time)
                else:
                    # for trans in self.net.transition():
                    #     modes = trans.modes(time=global_time)
                    #     if modes:
                    #         enabled_transitions[trans] = modes
                    #         sum_weight += trans.get_weight()
                    #
                    # if not enabled_transitions:
                    #     # increment time to next token's time!
                    #     last_time = global_time
                    #     if not times_to_check:
                    #         break
                    #     global_time = times_to_check.pop(0)
                    #     if TimeSimulator.DEBUG:
                    #         print "no transitions enabled at time {} checking model at time {}".format(last_time, global_time)
                    # else:

                    # print datetime.datetime.fromtimestamp(global_time).strftime('%Y-%m-%d %H:%M:%S')

                    # pick according to transition weights
                    trans_to_fire = random.uniform(0, sum_weight)

                    cumulative_weight = 0
                    transition = None
                    for enabled, modes in enabled_transitions.iteritems():
                        cumulative_weight += enabled.get_weight()
                        if trans_to_fire < cumulative_weight:
                            transition = enabled
                            break
                    modes = enabled_transitions[transition]
                    mode = None
                    if queuing_policy == QueueingPolicy.random:
                        # pick randomly from modes:
                        mode = random.choice(modes)
                    elif queuing_policy == QueueingPolicy.earliest_due_date:
                        # pick the ones with the lowest time stamps on patients queuing
                        earliest_mode = None
                        earliest_time = -1
                        for mode in modes:
                            tokens = mode.image()
                            token_time = -1
                            for token in tokens:
                                if is_patient(token):
                                    token_time = token.get_time() if token_time == -1 else min(token.get_time(), token_time)
                            is_earlier = earliest_time == -1 or token_time < earliest_time
                            if is_earlier:
                                earliest_time = token_time
                                earliest_mode = mode
                        mode = earliest_mode

                    time = transition.fire(mode, time=global_time, snapshot=self.snapshot_principle, net=self.net, tokens=self.sorted_tokens)
                    times_to_check.add(time)
                    if TimeSimulator.DEBUG:
                        print "times to check: {}".format(times_to_check)
        return self.net.get_marking()

    def init_help(self):
        return {
            "#trace": {
                "title": "Trace",
                "content": "the states and transitions explored so far"
            },
            "#model": {
                "title": "Model",
                "content": "the model being simulated"
            },
            "#alive .ui #ui-quit": {
                "title": "Stop",
                "content": "stop the simulator (server side)"
            },
            "#alive .ui #ui-help": {
                "title": "Help",
                "content": "show this help"
            },
            "#alive .ui #ui-about": {
                "title": "About",
                "content": "show information about the simulator"
            },
        }

    def collect_tokens_from_net(self):
        token_dict = defaultdict(list)
        for place in self.net.place():
            for token in place.tokens:
                if is_patient(token):
                    token_dict[token].append(place)
        return token_dict

    def parse_net_structure(self):
        """
        Collects forward pointers from places to transitions
        (showing which transitions depend on the tokens)
        """
        self.place_x_pat2transition = SortedDict({})
        for place in self.net.place():
            self.place_x_pat2transition[place.get_name()] = SortedDict({})
        for transition in self.net.transition():
            # todo need to get the guard on the transition and find the corresponding tokens
            add_all = str(transition.guard) == 'True'
            for place in transition.input():
                name = place[0].get_name()
                if 'all' not in self.place_x_pat2transition[name]:
                    self.place_x_pat2transition[name]['all'] = SortedSet([])
                for tok in self.sorted_tokens:
                    if add_all:
                        self.place_x_pat2transition[name]['all'].add(transition)
                    elif tok.value in str(transition.guard):
                        if tok.value not in self.place_x_pat2transition[name]:
                            self.place_x_pat2transition[name][tok.value] = SortedSet([])
                        self.place_x_pat2transition[name][tok.value].add(transition)
            # for place in transition.input():
            #     name = place[0].get_name()
            #     self.place_x_pat2transition[name].add(transition)

    def get_available_tokens_and_places(self, global_time):
        available_tokens_and_places = {}
        for token, place in self.sorted_tokens.iteritems():
            if isinstance(token, tuple):
                time = token[0].get_time()
            else:
                time = token.get_time()
            if time <= global_time and is_patient(token):
                available_tokens_and_places[token] = place
            #else:
            #    break
        return available_tokens_and_places

    # def assume_ordered_tokens(self):
    #     if len(self.sorted_tokens) > 1:
    #         last = self.sorted_tokens.keys()[0]
    #         for tok in self.sorted_tokens.keys():
    #             if tok < last:
    #                 print "error in sort order!"
    #             last = tok


def is_patient(token):
    """
    Checks whether the token represents a patient
    :param token: the token specifying a resource (patients start with "pat")
    :return: boolean
    """
    if isinstance(token, tuple):
        return token[0].type == TokenType.patient
    return token.type==TokenType.patient