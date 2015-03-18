# coding: utf-8
# An example model using the time extension plugin of SNAKES
from snakes.nets import *

import snakes.plugins
snakes.plugins.load("gv", "snakes.nets", "nets")
from nets import *


from snakes.plugins.time_cpn import Distribution, TimeSimulator
snakes.plugins.load('time_cpn', 'snakes.nets', 'time')
from time import Token, Transition, Place

t_joe1 = Token('patient_joe', time=2.4)
t_joe2 = Token('patient_joe', time=5.1)
t_joe3 = Token('patient_joe', time=5.1)

print t_joe1 < t_joe2

print t_joe1 == t_joe2
print t_joe3 == t_joe2


n = PetriNet('Hospital')
# joe needs a blood draw (BD) at time 2.4 and a blood test (BT) at time 5.1
t_joe1 = Token('patient_joe', time=2.4)
t_joe2 = Token('patient_joe', time=5.1)

# linda needs only a blood draw at time 3.0
t_lin1 = Token('patient_linda', time=3.4)

t_nurse = Token('nurse_bella', time=0.0)

pNurse = Place('pNurse', [])
pNurse.add(t_nurse)
n.add_place(pNurse)

# blood draw
pBD_queue = Place('BD_queue', [])
pBD_queue.add(t_joe1)
pBD_queue.add(t_lin1)
n.add_place(pBD_queue)
pBD_service = Place('BD_serv', [])
n.add_place(pBD_service)
pBD_finish = Place('BD_finish', [])
n.add_place(pBD_finish)

# wire patients for blood draw
n.add_transition(Transition('tBD_enter', guard=Expression("True"), dist=Distribution("exponential", rate=1)))
n.add_input('BD_queue', 'tBD_enter', Variable('pat'))
n.add_input('pNurse', 'tBD_enter', Variable('nurse'))
n.add_output('BD_serv', 'tBD_enter', Tuple([Expression('pat'), Expression('nurse')]))

n.add_transition(Transition('tBD_exit', guard=Expression("True"), dist=Distribution("immediate")))
n.add_input('BD_serv', 'tBD_exit', Tuple([Variable('pat'), Variable('nurse')]))
n.add_output('BD_finish', 'tBD_exit', Expression('pat'))
n.add_output('pNurse', 'tBD_exit', Expression('nurse'))

# infusion
pInf_queue = Place('Inf_queue', [])
n.add_place(pInf_queue)
pInf_service = Place('Inf_serv', [])
n.add_place(pInf_service)
pInf_finish = Place('Inf_finish', [])
n.add_place(pInf_finish)

# wire patients for infusion
n.add_transition(Transition('tInf_enter', guard=Expression("True"), dist=Distribution("exponential", rate=1)))
n.add_input('Inf_queue', 'tInf_enter', Variable('pat'))
n.add_input('pNurse', 'tInf_enter', Variable('nurse'))
n.add_output('Inf_serv', 'tInf_enter', Tuple([Expression('pat'), Expression('nurse')]))

n.add_transition(Transition('tInf_exit', guard=Expression("True"), dist=Distribution("immediate")))
n.add_input('Inf_serv', 'tInf_exit', Tuple([Variable('pat'), Variable('nurse')]))
n.add_output('Inf_finish', 'tInf_exit', Expression('pat'))
n.add_output('pNurse', 'tInf_exit', Expression('nurse'))

# add transition for joe to go from blood draw to infusion
n.add_transition(Transition('tBD_Inf', guard=Expression("pat=='patient_joe'"), dist=Distribution("scheduled", mintime=5.4)))
n.add_input('BD_finish', 'tBD_Inf', Variable('pat'))
n.add_output('Inf_queue', 'tBD_Inf', Expression('pat'))


#TODO add more transitions
n.draw("model.png")

# print n.transition('tBD_enter').modes(time=1)
# print n.transition('tBD_enter').modes(time=2)
# print n.transition('tBD_enter').modes(time=3)


sim = TimeSimulator(net=n)
sim.simulate_one()

n.draw("model2.png")

# tEnter = n.transition('tBD_enter')
# modes = tEnter.modes(time=3)
# tEnter.fire(modes[0], time=3)

#
# tExit = n.transition('tBD_exit')
#
# modes = tExit.modes(time=4)
# print modes
#
# print type(pBD_service.tokens)
#
# print tExit.modes(time=5)
# print tExit.modes(time=6)
# print tExit.modes(time=7)






#sim = TimeSimulator(n)

#sim.simulate_one(n.get_marking(), 0.5)
#n.draw("model2.png")


