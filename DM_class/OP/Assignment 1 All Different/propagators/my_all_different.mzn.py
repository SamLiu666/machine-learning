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
        self.args = args  # pass parametes
        self.anns = anns  # save answer
        #print("%%", type(self.args[0][0]))
        #print("%%", type(self.args), len(self.args))
        #print("%%", self.anns, len(self.anns))
        # Make sure 'wakeup' will be called whenever x is fixed.

        for numbers in self.args:
            #print("%%", numbers)
            for n in numbers:
                # print("%%", type(n)==int)
                if type(n) == int:
                     continue
                # if n.is_fixed():
                n.attach_event(Event.FIX, self.wakeup)  # wake up
                #print("%%", "wakeup")
            
    def wakeup(self, solver):
        print("%%", "wakeup")  # check for wake up
        self.queue(solver)
    
    def propagate(self, solver):
        # TODO
        #print("%%", type(solver))
        
        return True
