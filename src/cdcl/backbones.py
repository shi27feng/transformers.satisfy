import argparse
import time
from typing import Optional
from cdcl_heuristic import Tuple, CNFFormula
from cdcl_heuristic import cdcl
import heapq


def find_backbones(input_file: str) -> Tuple[Optional[list], int, float]:
    """
    Finds all backbones of the given DIMACS formula from the `input_file`.

    :param input_file: description of the DIMACS formula
    :return: a list of backbones literals in the case that the formula is SAT, none otherwise. Number of SAT solver
        calls and total CPU time
    """
    # Counter for number of SAT solver calls
    sat_calls = 0
    start_time = time.time()

    if input_file[-3:] == "cnf":
        input = open(input_file, mode="r")

    else:
        print("Unsupported file extension. File extension must be `.cnf` for DIMACS")
        cpu_time = time.time() - start_time
        return None, sat_calls, cpu_time

    dimacs_formula = input.read()
    dimacs_formula = dimacs_formula.splitlines()

    # This is the representation of the formula as a list of lists of literals
    formula = [list(map(int, clause[:-2].strip().split())) for clause in dimacs_formula if clause != "" and
               clause[0] not in ["c", "p", "%", "0"]]

    cnf_formula = CNFFormula(formula)
    sat, model, decisions, unit_propagations, restarts = cdcl(cnf_formula)
    sat_calls += 1

    # If the formula is UNSAT then there are no backbones and we can end the search
    if not sat:
        print("Formula is UNSAT")
        cpu_time = time.time() - start_time
        return None, sat_calls, cpu_time

    # Candidates are represented as lists of 2 numbers: `[priority, literal]`, where `priority` is the negative
    # number of clauses from `formula` containing this `literal`.
    # This `candidates` list represents priority queue (min heap) of literals ordered by the `priority`.
    # In some cases the literal can be discarded from the heap by setting the `literal` to 0, which we assume
    # is not a variable name.
    candidates = []
    for literal in model:
        count = 0
        for clause in formula:
            if literal in clause:
                count += 1

        candidates.append([-count, literal])

    heapq.heapify(candidates)

    # A list of backbones initially empty
    backbones = []

    # While the candidates heap is not empty, pop the literal with the minimum priority (which is negative number of
    # occurrences of this literal in the clauses so it is actually the literal with the highest occurrence) which is
    # not discarded, run CDCL on formula together with negation of this literal. If the result is UNSAT,
    # then the literal is backbone and can be added to the list of backbones and to the original formula as unit
    # clause. Otherwise, we can prune the current priority queue (heap) by removing those literals that are not
    # present in the current model of the formula. Removing of the literals from the min heap cannot be done
    # explicitly, therefore we set the value of the literal to 0 instead.
    while candidates:
        priority, literal = heapq.heappop(candidates)
        if literal == 0:
            continue

        cnf_formula = CNFFormula(formula + [[-literal]])
        sat, model, decisions, unit_propagations, restarts = cdcl(cnf_formula)
        sat_calls += 1

        if not sat:
            backbones.append(literal)
            formula.append([literal])

        else:
            temp = set(model)
            for c in candidates:
                if c[1] not in temp:
                    c[1] = 0

    cpu_time = time.time() - start_time

    # If there are any backbones, sort them by the variable name and return them.
    if backbones:
        backbones.sort(key=abs)
        return backbones, sat_calls, cpu_time

    else:
        return None, sat_calls, cpu_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file which contains a description of a DIMACS formula.")
    args = parser.parse_args()

    backbones_list, sat_solver_calls, total_cpu_time = find_backbones(args.input)

    if backbones_list:
        print("Backbones =", backbones_list)

    else:
        print("There are no backbones.")

    print("CPU time =", total_cpu_time, "seconds")
    print("Number of SAT solver calls =", sat_solver_calls)
