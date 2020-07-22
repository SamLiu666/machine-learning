"""
All-different propagator
"""

from sprue.propagator import Propagator
from sprue.var import IntVar, Event, ConstView


def register_handlers(registry):
    #print("%% It lives!")
    registry.register_propagator('my_all_different', AllDiffProp)

class AllDiffProp(Propagator):
    def __init__(self, solver, args, anns):
        # TODO
        super().__init__(self)
        self.args = args  # pass parametes list
        self.anns = anns  # save answer list
        self.count = 0
        # print("%%", type(self.args[0][0]))
        #print("%%", type(self.args), len(self.args))
        #print("%%", self.anns, len(self.anns))
        # Make sure 'wakeup' will be called whenever x is fixed.
        for numbers in self.args:
            #print("%%", numbers)  # numbers list
            for n in numbers:
                # print("%%", type(n)==int)
                if type(n) == int:
                     continue
                # if n.is_fixed():
                n.attach_event(Event.FIX, self.wakeup)  # wake up
                #print("%%", "wakeup")
            
    def wakeup(self, solver):
        #print("%%", "wakeup")  # check for wake up
        self.queue(solver)

    def propagate(self, solver):

        for numbers in self.args:
            #print("%%", numbers)  # numbers list
            for number in numbers:
                if type(number)!=int:

                    for n in numbers:
                        # print("%%", type(n)==int)
                        if type(n) != int:
                            if number.is_fixed() and not n.is_fixed():
                                n.remove_value(solver, number.value())
                    


    # def propagate(self, solver):
    #     # TODO
    #     # .1 how can I remove the int value and intVAR value 
    #     #print("%%", self.args) # list of 9 list including int and intVar

    #     for number in self.args:
    #         # 1. do the first loop for one 
    #         #print("%%", number)  # number is list

    #         self.count += 1
    #         temp = set() #
    #         #print("%% the number loop: ", self.count, self.anns)

    #         for n in number:  # n is int or intVar

    #             # if n is int, add it into exit for deleting
    #             if type(n) == int:
    #                 # temp.append(n)  # save the int value
    #                 temp.add(n)

    #             elif n.is_fixed():
    #                 # 2. if n is_fixed, add it into exit for deleting
    #                 #  n.domain() is set
    #                 #print("%% n -varInt: ", n.domain(), type(n.domain()))
    #                 #para = n.in_domain()
    #                 # continue
    #                 para = list(n.domain())
    #                 temp.add(n)
    #                 #temp.append(para)  
    #                 #print("%%", n._domain, para)
                    

    #             else:
    #                 # 3. remove all the exits value temp from other domain

    #                 # pass
    #                 if temp:
                        
    #                     #print("%% 1. temporary: ", temp, type(temp))  # temp is list of int and intVAR
    #                     tt = list(temp)
    #                     for t in tt:
    #                         #print("%%", type(t)) # int and intVar
    #                         if type(t) == int:
    #                             n.remove_value(solver, t)
    #                             print("%%2. delte: ", t)
    #                             print("%%n domains ==: ", n)

    #                         # elif n.in_domain(t.domain()):
    #                         else:
    #                             p = t.domain()
    #                             print("%% p= ", p)
    #                             n.remove_value(solver, p)
    #                             print("%%2.2. delte: ", p)
    #                             print("%%n domains ==: ", n)

            #print("%%temporary: ", temp) 

            # print("%%temporary: ",temp)  