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
        self.args = args[0]  # pass parametes list, for just one domain important
        self.anns = anns  # save answer list
        # print("%%", type(self.args[0][0]))
        #print("%%", type(self.args), len(self.args))
        #print("%%", self.anns, len(self.anns))
        # Make sure 'wakeup' will be called whenever x is fixed.
        for numbers in self.args:
            #print("%%", numbers)  # numbers list
            # print("%%", type(n)==int)
            """part A: when x is fixed, wake up"""
            if type(numbers) != int:
                numbers.attach_event(Event.FIX, self.wakeup)  # wake up
                # if n.is_fixed():
                #print("%%", "wakeup")
            
    def wakeup(self, solver):
        #print("%%", "wakeup")  # check for wake up
        self.queue(solver)

    def propagate(self, solver):
        # sudo
        ans = set([])  # save for values
        count = 0   # count times


        for i in range(0, 9):
            if type(self.args[i]) == int:
                for j in range(0, 9):
                    if i!=j and type(self.args[j]) != int:
                        # remove the fixed value from other domains
                        self.args[j].remove_value(solver, self.args[i])

        for i in range(0, 9):
            for j in range(0, 9):
                if i!=j and type(self.args[i]) != int and self.args[i].is_fixed():
                    if type(self.args[j]) == IntVar and self.args[j].is_fixed():  # For intVar
                        if self.args[j].value() == self.args[i].value():
                            return False
                    if type(self.args[j]) == int:
                        if self.args[j] == self.args[i].value():
                            return False

        for i in range(0, 9):
            if type(self.args[i]) != int and self.args[i].is_fixed():
                for j in range(0, 9):
                    if i!=j and type(self.args[j]) != int:
                        # remove the fixed args[j] value
                        self.args[j].remove_value(solver, self.args[i].value())

        for i in range(0, 9):
            # count the domian numbers
            if type(self.args[i]) != int and len(self.args[i].domain()) > 1:
                count += 1
                ans = ans | set(self.args[i].domain())
            if len(ans) < count:
                return False
        return True
                    


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