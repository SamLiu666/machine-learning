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
        # TODO
        # .1 how can I remove the int value and intVAR value 

        temp= 0
        for number in self.args:
            #print("%%", number)  # number is list

            self.count += 1
            temp = []
            print("%% the number loop: ", self.count, self.anns)

            for n in number:  # n is int or intVar

                # if n is int, add it into exit for deleting
                if type(n) == int:
                    temp.append([n])

                elif n.is_fixed():
                    # if n is_fixed, add it into exit for deleting
                    #print("%%", n._domain)
                    #para = n.in_domain()
                    #print("%%", n._domain, para)
                    temp.append(n)

                else:
                    # remove the exits value from domain
                    if n.in_domain(temp[0]):
                        n.remove_value(solver, 6)
                        print("%% values: ", n.domain())

            print("%%temporary: ",temp)

                
                    #print("%%", n)
           
            # else:
            #     print("%%", number[0])
                # if number[0].is_fixed():
                #     self.anns.append(number[0])
                # else:
                #     number[0].remove_value(solver, number[0].value())  # wake up
                            # self.anns.append(n.value)
            