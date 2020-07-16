"""
All-different propagator
"""

from sprue.propagator import Propagator
from sprue.var import IntVar, Event, ConstView


def register_handlers(registry):
    print("%% It lives!")
    registry.register_propagator('my_all_different', AllDiffProp)

class AllDiffProp(Propagator):
    def __init__(self, solver, args, anns):
        # TODO
        super().__init__(self)
        self.args = args
        # Make sure 'wakeup' will be called whenever x is fixed.

            
    def wakeup(self, solver):
        self.queue(solver)
    
    def propagate(self, solver):
        # TODO
        
        return True
