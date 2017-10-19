import re
from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list`
        property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # create concrete Action objects based on the domain action schema for: Load, Unload, and
        # Fly concrete actions definition: specific literal action that does not include variables
        # as with the schema for example, the action schema 'Load(c, p, a)' can represent the
        # concrete actions 'Load(C1, P1, SFO)' or 'Load(C2, P2, JFK)'.  The actions for the
        # planning problem must be concrete because the problems in forward search and Planning
        # Graphs must use Propositional Logic.
        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            # create all load ground actions from the domain Load action
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [expr("At({}, {})".format(cargo, airport)), \
                            expr("At({}, {})".format(plane, airport))]#, \
                            #expr("Cargo({})".format(cargo)), expr("Plane({})".format(plane)), \
                            #expr("Airport({})".format(airport))]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(cargo, plane))]
                        effect_rem = [expr("At({}, {})".format(cargo, airport))]
                        load = Action(expr("Load({}, {}, {})".format(cargo, plane, airport)), \
                            [precond_pos, precond_neg], [effect_add, effect_rem])
                        loads.append(load)
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # create all Unload ground actions from the domain Unload action
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [expr("In({}, {})".format(cargo, plane)), \
                            expr("At({}, {})".format(plane, airport))] #, \
                            #expr("Cargo({})".format(cargo)), expr("Plane({})".format(plane)), \
                            #expr("Airport({})".format(airport))]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(cargo, plane))]
                        effect_rem = [expr("At({}, {})".format(cargo, airport))]
                        unload = Action(expr("Unload({}, {}, {})".format(cargo, plane, airport)), \
                            [precond_pos, precond_neg], [effect_add, effect_rem])
                        unloads.append(unload)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for departure in self.airports:
                for arrival in self.airports:
                    if departure != arrival:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, departure))]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, arrival))]
                            effect_rem = [expr("At({}, {})".format(p, departure))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, departure, arrival)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # implement
        possible_actions = []
        kbase = PropKB()
        kbase.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kbase.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kbase.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        #print_actions(possible_actions)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # implement
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kbase = PropKB()
        kbase.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kbase.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        # without preconditions, the distance is just the number of actions that will satisfy
        # all goals, which is just the number of goals, assuming no action can satisfy multiple
        # goals, which is the case for air cargo.
        return len(self.goal)        

def print_actions(actions):
    """ Print out actions and their args """
    for action in actions:
        print()
        print("Action, Args: {}, {}".format(action.name, action.args))

def print_node(node, problem):
    """ Print out a node """
    print()
    print("Node state: {}, action: {}".format(node.state, node.action))
    '''
    print("Node expand:")
    expand_list = node.expand(problem)
    for child in expand_list:
        print_node(child, problem)
    '''

def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           #expr('Cargo(C1)'),
           #expr('Cargo(C2)'),
           #expr('Plane(P1)'),
           #expr('Plane(P2)'),
           #expr('Airport(JFK)'),
           #expr('Airport(SFO)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


CARGO_LOCATIONS = None
PLANE_LOCATIONS = None

def assign_locations(positive_preconditions, cargos, planes, airports):
    """ Assign locations based in on positive preconditions """
    global CARGO_LOCATIONS, PLANE_LOCATIONS

    CARGO_LOCATIONS = dict()
    PLANE_LOCATIONS = dict()

    for expression in positive_preconditions:
        _, item1, item2 = re.findall(r"[\w]+", str(expression))
        #print("Items: {}, {}".format(item1, item2))
        if item1 in cargos:
            CARGO_LOCATIONS[item1] = item2
        if item1 in planes:
            PLANE_LOCATIONS[item1] = item2

def find_negatives(positive_preconditions, cargos, planes, airports):
    """ Find all negative preconditions given positive preconditions """
    assign_locations(positive_preconditions, cargos, planes, airports)
    neg = []
    for cargo in cargos:
        location = CARGO_LOCATIONS[cargo]
        for airport in airports:
            if location != airport:
                neg.append(expr('At({}, {})'.format(cargo, airport)))
        for plane in planes:
            if location != plane:
                neg.append(expr('In({}, {})'.format(cargo, plane)))
    for plane in planes:
        location = PLANE_LOCATIONS[plane]
        for airport in airports:
            if location != airport:
                neg.append(expr('At({}, {})'.format(plane, airport)))
    return neg

def air_cargo_p2() -> AirCargoProblem:
    # implement Problem 2 definition
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)')]
    neg = find_negatives(pos, cargos, planes, airports)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')]
    neg = find_negatives(pos, cargos, planes, airports)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)
