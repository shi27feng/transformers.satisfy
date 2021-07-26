import argparse
import random

from formula2cnf import formula2cnf
import time
from typing import Tuple, Optional
from collections import deque
import numpy as np

"""
CDCL with decision heuristic:
the implementation by https://github.com/jaras209/SAT_solver
"""


class Clause:
    """
    The class which represents a single clause.
    """

    def __init__(self, literals, w1=None, w2=None, learned=False, lbd=0):
        self.literals = literals  # describes how exactly does this clause look like
        self.size = len(self.literals)
        self.w1 = w1
        self.w2 = w2
        self.learned = learned
        self.lbd = lbd

        if (not w1) and (not w2):
            if len(self.literals) > 1:
                self.w1 = 0
                self.w2 = 1

            elif len(self.literals) > 0:
                self.w1 = self.w2 = 0

    def partial_assignment(self, assignment: list) -> list:
        """
        Performs partial assignment of this clause with given `assignment` and returns the resulting list of literals,
        i.e. if the clause is SAT then returns empty list, otherwise returns the remaining list of unassigned literals.
        (It it currently used only in the heuristic selection of decision literal: `get_decision_literal`)

        :param assignment: the assignment
        :return: if the clause is SAT then returns empty list, otherwise returns the remaining list of unassigned
        literals
        """
        unassigned = []
        for literal in self.literals:
            if assignment[abs(literal)] == literal:
                return []

            if assignment[abs(literal)] == 0:
                unassigned.append(literal)

        return list(unassigned)

    def update_watched_literal(self, assignment: list, new_variable: int) -> Tuple[bool, int, Optional[int]]:
        """
        Updates the watched literal of this Clause given the assignment `assignment` and the latest assigned variable
        `new_variable` which is used to update the watched literal, if necessary.

        :param new_variable: name of the variable which was currently changed
        :param assignment: a current assignment list
        :return: Tuple `(success, new_watched_literal, unit_clause literal)` where `success` represents whether the
        update was successful or the Clause is unsatisfied, `new_watched_literal` is the new watched literal,
        `unit_clause_literal` represent the unit clause literal in the case that the Clause becomes unit during the
        update of the watched literal.
        """

        # Without loss of generality, the old watched literal index, that we need to change, is `self.w1`
        if new_variable == abs(self.literals[self.w2]):
            temp = self.w1
            self.w1 = self.w2
            self.w2 = temp

        # If Clause[self.w1] is True in this new variable assignment or
        # Clause[self.w2] has been True previously, then the Clause is satisfied
        if (self.literals[self.w1] == assignment[abs(self.literals[self.w1])] or
                self.literals[self.w2] == assignment[abs(self.literals[self.w2])]):
            return True, self.literals[self.w1], False

        # If Clause[self.w1] is False in this new variable assignment and
        # Clause[self.w2] is also False from previous assignment, then the Clause is unsatisfied
        if (-self.literals[self.w1] == assignment[abs(self.literals[self.w1])] and
                -self.literals[self.w2] == assignment[abs(self.literals[self.w2])]):
            return False, self.literals[self.w1], False

        # If Clause[self.w1] is False in this new variable assignment and
        # Clause[self.w2] is still unassigned, then look for new index of the watched literal `self.w1`
        if (-self.literals[self.w1] == assignment[abs(self.literals[self.w1])] and
                assignment[abs(self.literals[self.w2])] == 0):
            old_w1 = self.w1
            for w in [(self.w1 + i) % self.size for i in range(self.size)]:
                # new index `w` must not be equal to `self.w2` and
                # Clause[w] cannot be False in the current assignment
                if w == self.w2 or -self.literals[w] == assignment[abs(self.literals[w])]:
                    continue

                self.w1 = w
                break

            # If the new watched literal index `self.w1` has not been found then the Clause is unit with
            # Clause[self.w2] being the only unassigned literal.
            if self.w1 == old_w1:
                return True, self.literals[self.w1], True

            # Otherwise the state of the Clause is either not-yet-satisfied or satisfied -> both not important
            return True, self.literals[self.w1], False

    def is_satisfied(self, assignment: list) -> bool:
        """
        (It it currently used only in the heuristic selection of decision literal: `get_decision_literal`)
        :param: assignment: the assignment list
        :return: True if the clause is satisfied in the `assignment`, i.e. one of its watched literals is True.
        """
        return (self.literals[self.w1] == assignment[abs(self.literals[self.w1])] or
                self.literals[self.w2] == assignment[abs(self.literals[self.w2])])


class CNFFormula:
    """
    The class which represents one formula in CNF.
    """

    def __init__(self, formula):
        self.formula = formula  # list of lists of literals
        self.clauses = [Clause(literals) for literals in self.formula]  # list of clauses
        self.learned_clauses = []
        self.variables = set()  # unordered unique set of variables in the formula
        self.watched_lists = {}  # dictionary: list of clauses with this `key` literal being watched
        self.unit_clauses_queue = deque()  # queue for unit clauses
        self.assignment_stack = deque()  # stack for representing the current assignment for backtracking
        self.assignment = None  # the assignment list with `variable` as index and `+variable/-variable/0` as values
        self.antecedent = None  # the antecedent list with `variable` as index and `Clause` as value
        self.decision_level = None  # the decision level list with `variable` as index and `decision level` as value
        self.positive_literal_counter = None
        self.negative_literal_counter = None

        for clause in self.clauses:
            # If the clause is unit right at the start, add it to the unit clauses queue
            if clause.w1 == clause.w2:
                self.unit_clauses_queue.append((clause, clause.literals[clause.w2]))

            # For every literal in clause:
            for literal in clause.literals:
                variable = abs(literal)
                # - add variable to the set of all variables
                self.variables.add(variable)

                # - Create empty list of watched clauses for this variable, if it does not exist yet
                if variable not in self.watched_lists:
                    self.watched_lists[variable] = []

                # - Update the list of watched clauses for this variable
                if clause.literals[clause.w1] == literal or clause.literals[clause.w2] == literal:
                    if clause not in self.watched_lists[variable]:
                        self.watched_lists[variable].append(clause)

        # Set the assignment/antecedent/decision_level list of the Formula with initial values for each variable
        max_variable = max(self.variables)
        self.assignment = [0] * (max_variable + 1)
        self.antecedent = [None] * (max_variable + 1)
        self.decision_level = [-1] * (max_variable + 1)
        self.positive_literal_counter = np.zeros((max_variable + 1), dtype=np.float64)
        self.negative_literal_counter = np.zeros((max_variable + 1), dtype=np.float64)

    def all_variables_assigned(self) -> bool:
        """
        :return: True if the formula is satisfied, i.e. if all variables are assigned
        """
        return len(self.variables) == len(self.assignment_stack)

    def assign_literal(self, literal: int, decision_level: int) -> Tuple[bool, Optional[Clause]]:
        """
        Assigns the literal at the specified decision level.

        :param decision_level: decision level of the literal
        :param literal: literal to be assigned
        :return: A tuple `(succeeded, antecedent_of_conflict)` where `succeeded` is `True` if the assignment was
            successful and False otherwise, `antecedent_of_conflict` is a unsatisfied conflict clause. I.e.
            `(succeeded, antecedent_of_conflict)` = `(True, None)` if the partial assignment did not derive any conflict.
            `(succeeded, antecedent_of_conflict)` = `(False, clause)` if the partial assignment derived unsatisfied
            clause `clause`.
        """
        # Add literal to assignment stack and set the value of corresponding variable in the assignment list
        self.assignment_stack.append(literal)
        self.assignment[abs(literal)] = literal
        self.decision_level[abs(literal)] = decision_level

        # Copy the watched list of this literal because we need to delete some of the clauses from it during
        # iteration and that cannot be done while iterating through the same list
        watched_list = self.watched_lists[abs(literal)][:]

        # For every clause in the watched list of this variable perform the update of the watched literal and
        # find out which clauses become unit and which become unsatisfied in the current assignment
        for clause in watched_list:
            success, watched_literal, unit = clause.update_watched_literal(self.assignment, abs(literal))

            # If the clause is not unsatisfied:
            if success:
                # If the watched literal was changed:
                if abs(watched_literal) != abs(literal):
                    # Add this clause to the watched list of the new watched literal
                    if clause not in self.watched_lists[abs(watched_literal)]:
                        self.watched_lists[abs(watched_literal)].append(clause)

                    # Remove this clause from the watched list of the old watched literal
                    self.watched_lists[abs(literal)].remove(clause)

                # If the clause is unit then add the clause to the unit clauses queue
                if unit:
                    if clause.literals[clause.w2] not in [x[1] for x in self.unit_clauses_queue]:
                        self.unit_clauses_queue.append((clause, clause.literals[clause.w2]))

            # If the clause is unsatisfied return False
            if not success:
                return False, clause

        return True, None

    def backtrack(self, decision_level: int) -> None:
        """
        Delete the assignment stack up until the `decision_level`,
        i.e. assignment of all variables with decision level > `decision_level` will be removed.

        :param decision_level: specify the decision level
        """
        while self.assignment_stack and self.decision_level[abs(self.assignment_stack[-1])] > decision_level:
            literal = self.assignment_stack.pop()
            self.assignment[abs(literal)] = 0
            self.antecedent[abs(literal)] = None
            self.decision_level[abs(literal)] = -1

    @staticmethod
    def resolve(clause1: list, clause2: list, literal: int) -> list:
        """
        Compute the resolvent of the clauses from the arguments with respect to the literal in the argument.

        :param clause1: first clause which contains `-literal`
        :param clause2: second clause which contains `literal`
        :param literal: literal which is used for resolution
        :return: a list of literals representing the resolvent
        """
        in_clause1 = set(clause1)
        in_clause2 = set(clause2)
        in_clause1.remove(-literal)
        in_clause2.remove(literal)
        return list(in_clause1.union(in_clause2))

    def conflict_analysis(self, antecedent_of_conflict: Clause, decision_level: int) -> int:
        """
        Consists of analyzing the most recent conflict, learning a new clause (assertive clause) from the conflict
        and computing the backtracking level (assertive level), which is the second highest decision level in the
        assertive clause.

        :param antecedent_of_conflict: a conflict clause which was derived
        :param decision_level: the current decision level where the conflict was derived
        :return: -1 if a conflict at decision level 0 is detected (which implies that the formula is unsatisfiable).
            Otherwise, a decision level which the solver should backtrack to.
        """
        # If the conflict was detected at decision level 0, return -1
        if decision_level == 0:
            return -1

        # Find the literals of the assertive clause
        assertive_clause_literals = antecedent_of_conflict.literals
        current_assignment = deque(self.assignment_stack)
        while len([l for l in assertive_clause_literals if self.decision_level[abs(l)] == decision_level]) > 1:
            while True:
                literal = current_assignment.pop()
                if -literal in assertive_clause_literals:
                    assertive_clause_literals = self.resolve(assertive_clause_literals,
                                                             self.antecedent[abs(literal)].literals, literal)
                    break

        # Find the assertion level and the unit literal of the assertive clause which will be the only
        # unassigned literal of the assertive clause after backtrack to assertion level.
        # Also find out the `w2` index for the assertive clause which is the index of that unassigned literal.
        # Also find out which decision levels are present in the assertive clause. This will be used for
        # finding out LBD of the assertive clause.
        # Lastly, decay the counters of the literals for VSIDS heuristic and add 1 to counters of all those literals
        # which appear in this new clause.
        assertion_level = 0
        unit_literal = None
        w2 = None
        decision_level_present = [False] * (decision_level + 1)
        for index, literal in enumerate(assertive_clause_literals):
            if assertion_level < self.decision_level[abs(literal)] < decision_level:
                assertion_level = self.decision_level[abs(literal)]

            if self.decision_level[abs(literal)] == decision_level:
                unit_literal = literal
                w2 = index

            if not decision_level_present[self.decision_level[abs(literal)]]:
                decision_level_present[self.decision_level[abs(literal)]] = True

            self.positive_literal_counter = self.positive_literal_counter * 0.9
            self.negative_literal_counter = self.negative_literal_counter * 0.9
            if literal > 0:
                self.positive_literal_counter[literal] += 1

            else:
                self.negative_literal_counter[(abs(literal))] += 1

        # Find out LBD of the assertive clause
        lbd = sum(decision_level_present)

        # Find the `w1` index for the assertive clause which is the index of the last assigned literal
        # in the assertive clause with decision level equal to the assertion level
        w1 = None
        if len(assertive_clause_literals) > 1:
            current_assignment = deque(self.assignment_stack)
            found = False
            while current_assignment:
                literal = current_assignment.pop()
                if self.decision_level[abs(literal)] == assertion_level:
                    for index, clause_literal in enumerate(assertive_clause_literals):
                        if abs(literal) == abs(clause_literal):
                            w1 = index
                            found = True
                            break

                if found:
                    break

        else:
            w1 = w2

        # Create the assertive clause and update the watched lists of the watched literals
        assertive_clause = Clause(assertive_clause_literals, w1=w1, w2=w2, learned=True, lbd=lbd)
        self.watched_lists[abs(assertive_clause.literals[assertive_clause.w1])].append(assertive_clause)
        if assertive_clause.w1 != assertive_clause.w2:
            self.watched_lists[abs(assertive_clause.literals[assertive_clause.w2])].append(assertive_clause)

        # Add the assertive clause into the list of learned clauses
        self.learned_clauses.append(assertive_clause)

        # Clear the unit clauses queue and add the assertive clause into the unit clauses queue
        # together with its unit literal
        self.unit_clauses_queue.clear()
        self.unit_clauses_queue.append((assertive_clause, unit_literal))

        return assertion_level

    def unit_propagation(self, decision_level: int) -> Tuple[list, Optional[Clause]]:
        """
        Performs a unit propagation of this formula.

        :param decision_level: decision level
        :return: a tuple (assignment, antecedent_of_conflict) with assignment containing literals derived by unit
            propagation and antecedent_of_conflict which is either None, if the unit propagation was successful
            and no conflict was derived, or conflict clause.
        """
        propagated_literals = []
        while self.unit_clauses_queue:
            unit_clause, unit_clause_literal = self.unit_clauses_queue.popleft()
            propagated_literals.append(unit_clause_literal)
            self.antecedent[abs(unit_clause_literal)] = unit_clause

            success, antecedent_of_conflict = self.assign_literal(unit_clause_literal, decision_level)
            if not success:
                return propagated_literals, antecedent_of_conflict

        return propagated_literals, None

    def unassigned_heuristic(self) -> int:
        """
        Finds the unassigned literal which occurs in the largest number of not satisfied clauses.

        :return: the decision literal
        """
        number_of_clauses = -1
        decision_literal = None
        for variable in self.variables:
            if self.assignment[variable] == 0:
                positive_clauses = 0
                negative_clauses = 0
                for clause in self.watched_lists[variable]:
                    if not clause.is_satisfied(self.assignment):
                        unassigned = clause.partial_assignment(self.assignment)
                        if variable in unassigned:
                            positive_clauses += 1

                        if -variable in unassigned:
                            negative_clauses += 1
                if positive_clauses > number_of_clauses and positive_clauses > negative_clauses:
                    number_of_clauses = positive_clauses
                    decision_literal = variable

                if negative_clauses > number_of_clauses:
                    number_of_clauses = negative_clauses
                    decision_literal = -variable

        return decision_literal

    def vsids_heuristic(self) -> int:
        """
        Finds the unassigned literal based on VSIDS heuristic, i.e. the literal which is present the most in the
        learned clauses.

        :return: the decision literal
        """
        decision_literal = None
        best_counter = 0
        for variable in self.variables:
            if self.assignment[variable] == 0:
                if self.positive_literal_counter[variable] > best_counter:
                    decision_literal = variable
                    best_counter = self.positive_literal_counter[variable]

                if self.negative_literal_counter[variable] >= best_counter:
                    decision_literal = -variable
                    best_counter = self.negative_literal_counter[variable]

        return decision_literal

    def random_heuristic(self) -> int:
        """
        Finds the unassigned literal at random.

        :return: the decision literal
        """
        unassigned = []
        for variable in self.variables:
            if self.assignment[variable] == 0:
                unassigned.append(variable)

        decision_variable = random.choice(unassigned)

        if random.random() <= 0.5:
            return decision_variable

        else:
            return -decision_variable

    def pick_decision_literal(self, heuristic: int) -> int:
        """
        Pick an unassigned decision literal based on heuristic specified in the argument.

        :param heuristic: specifies a decision heuristic: `0`, `1` or `2`
        :return: a new decision literal
        """
        if heuristic == 0:
            return self.unassigned_heuristic()

        if heuristic == 1:
            return self.vsids_heuristic()

        if heuristic == 2:
            return self.random_heuristic()

    def delete_learned_clauses_by_lbd(self, lbd_limit: float) -> None:
        """
        Removes the learned clauses with lower LBD then the limit.
        :param lbd_limit: maximum LBD of the clause
        """

        lbd_limit = int(lbd_limit)
        new_learned_clauses = []
        for clause in self.learned_clauses:
            if clause.lbd > lbd_limit:
                self.watched_lists[abs(clause.literals[clause.w1])].remove(clause)
                if clause.w1 != clause.w2:
                    self.watched_lists[abs(clause.literals[clause.w2])].remove(clause)

            else:
                new_learned_clauses.append(clause)

        self.learned_clauses = new_learned_clauses

    def restart(self) -> None:
        """
        Performs the restart by clearing the unit clauses queue and backtracking to decision level 0.
        """
        self.unit_clauses_queue.clear()
        self.backtrack(decision_level=0)

    def print(self) -> None:
        """
        Prints basic information about the formula.
        """
        # Not used in the dpll program itself.
        print("Formula: ")
        print(self.formula)
        print("Clauses: ")
        for clause in self.clauses:
            print(clause.literals)

        print("Variables: ")
        print(self.variables)
        print("Watched lists: ")
        for variable, adj_list in self.watched_lists.items():
            print(variable, ": ")
            for clause in adj_list:
                print(clause.literals)


def cdcl(cnf_formula: CNFFormula,
         assumption: Optional[list] = None,
         heuristic: int = 1,
         conflicts_limit: int = 100,
         lbd_limit: int = 3) -> Tuple[bool, list, int, int, int]:
    """
    CDCL algorithm for deciding whether the DIMACS CNF formula in the argument `cnf_formula` is satisfiable (SAT) or
    unsatisfiable (UNSAT). In the case of SAT formula, the function also returns a model.

    :param cnf_formula: DIMACS CNF formula
    :param heuristic: Specifies a decision heuristic: `0`, `1` or `2`
    :param assumption: a list of integers representing assumption about the initial values of specified variables
    :param lbd_limit: a limit for LBD
    :param conflicts_limit: a limit for number of conflicts before a restart is used
    :return: a tuple (sat, model, decisions, unit_propagations, restarts) which describes whether the formula is SAT,
             what is its model, how many decisions were made during the derivation of the model, how many literals
             were derived by unit propagation and how many restarts were used
        """
    # Counters for number of decisions, unit propagations
    decision_level = 0
    decisions = 0
    unit_propagations = 0
    restarts = 0
    conflicts = 0

    # Unit propagation
    propagated_literals, antecedent_of_conflict = cnf_formula.unit_propagation(decision_level)
    unit_propagations += len(propagated_literals)

    if antecedent_of_conflict:
        return False, [], decisions, unit_propagations, restarts

    # Reverse the assumption list in order to pop elements from back which is done in O(1), because
    # popping the first element from the list is expensive.
    if assumption:
        assumption.reverse()

    while not cnf_formula.all_variables_assigned():
        # Find the literal for decision by either picking one from assumption or finding one using decision heuristic
        if assumption:
            decision_literal = assumption.pop()

        else:
            decision_literal = cnf_formula.pick_decision_literal(heuristic)

        decision_level += 1

        # Perform the partial assignment of the formula with the decision literal
        cnf_formula.assign_literal(decision_literal, decision_level)
        decisions += 1

        # Unit propagation
        propagated_literals, antecedent_of_conflict = cnf_formula.unit_propagation(decision_level)
        unit_propagations += len(propagated_literals)

        while antecedent_of_conflict:
            conflicts += 1

            # If the amount of conflicts reached the limit, perform restart and delete learned clauses with big LBD
            if conflicts == conflicts_limit:
                conflicts = 0
                conflicts_limit = int(conflicts_limit * 1.1)
                lbd_limit = lbd_limit * 1.1
                restarts += 1
                decision_level = 0
                cnf_formula.restart()
                cnf_formula.delete_learned_clauses_by_lbd(lbd_limit)
                break

            # Analyse conflict: learn new clause from the conflict and find out backtrack decision level
            backtrack_level = cnf_formula.conflict_analysis(antecedent_of_conflict, decision_level)
            if backtrack_level < 0:
                return False, [], decisions, unit_propagations, restarts

            # Backtrack
            cnf_formula.backtrack(backtrack_level)
            decision_level = backtrack_level

            # Unit propagation of the learned clause
            propagated_literals, antecedent_of_conflict = cnf_formula.unit_propagation(decision_level)
            unit_propagations += len(propagated_literals)

    return True, list(cnf_formula.assignment_stack), decisions, unit_propagations, restarts


def find_model(input_file: str, assumption: Optional[list] = None, heuristic: int = 1, conflicts_limit: int = 100,
               lbd_limit: int = 3) -> Optional[Tuple[bool, list, float, int, int, int]]:
    """
    Finds the model of the SAT formula from the `input_file` or returns `UNSAT`.

    :param input_file: describes the input formula. The file can contain either CNF formula in DIMACS format and in
                       that case ends with ".cnf" extension, or NNF formula in simplified SMT-LIB format and ends with
                        ".sat" extension.
    :param heuristic: specifies a decision heuristic: `0` - pick the unassigned literal which occurs in the largest
        number of not satisfied clauses, `1` - pick the unassigned literal based on VSIDS heuristic,
        `2` - pick the random unassigned literal
    :param assumption: a list of integers representing assumption about the initial values of specified variables
    :param conflicts_limit: a limit for number of conflicts before a restart is used
    :param lbd_limit: a limit for LBD
    :return: a tuple (sat, model, cpu_time, decisions, unit_propagations, restarts) which describes whether the formula
        is SAT or UNSAT, what is its model, how long the computation took, number of decisions, number of literals
        derived by unit propagation and number of restarts
    """
    if input_file[-3:] == "sat":
        formula2cnf(input_file=input_file, output_file=input_file[:-4] + ".cnf", left_to_right=True)
        input = open(input_file[:-4] + ".cnf", mode="r")

    elif input_file[-3:] == "cnf":
        input = open(input_file, mode="r")

    else:
        print("Unsupported file extension. File extension must be `.cnf` for DIMACS, or `.sat` for the simplified "
              "SMT-LIB format.")
        return

    dimacs_formula = input.read()
    dimacs_formula = dimacs_formula.splitlines()

    formula = [list(map(int, clause[:-2].strip().split())) for clause in dimacs_formula if clause != "" and
               clause[0] not in ["c", "p", "%", "0"]]

    cnf_formula = CNFFormula(formula)
    start_time = time.time()
    sat, model, decisions, unit_propagations, restarts = cdcl(cnf_formula, assumption, heuristic, conflicts_limit,
                                                              lbd_limit)
    cpu_time = time.time() - start_time
    if sat:
        model.sort(key=abs)
        print("SAT")
        print("Model =", model)
        print("Possible missing literals can have arbitrary value.")

    else:
        print("UNSAT")

    print("Total CPU time =", cpu_time, "seconds")
    print("Number of decisions =", decisions)
    print("Number of steps of unit propagation =", unit_propagations)
    print("Number of restarts =", restarts)

    return sat, model, cpu_time, decisions, unit_propagations, restarts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file which contains a description of a formula.")
    parser.add_argument("--assumption", type=int, default=None, nargs="+", help="A space separated sequence of "
                                                                                "integers representing assumption "
                                                                                "about initial values of specified "
                                                                                "variables")
    parser.add_argument("--heuristic", type=int, default=1, help="Specify a decision heuristic: `0` - pick the "
                                                                 "unassigned literal which occurs in the largest "
                                                                 "number of not satisfied clauses, `1` - pick the "
                                                                 "unassigned literal based on VSIDS heuristic, "
                                                                 "`2` - pick the random unassigned literal")
    parser.add_argument("--conflicts_limit", default=100, help="The initial limit on the number of conflicts before "
                                                               "the CDCL solver restarts")
    parser.add_argument("--lbd_limit", default=3, help="The initial limit on the number of different decision levels "
                                                       "in the learned clause for clause deletion")
    args = parser.parse_args()

    find_model(args.input, args.assumption, args.heuristic, args.conflicts_limit, args.lbd_limit)
