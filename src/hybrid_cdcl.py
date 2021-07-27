from typing import Tuple, Optional

import torch

from cdcl.cdcl_heuristic import CNFFormula
from src.conflict import vars_of_clauses, conflict_clauses
from src.loss import smooth_max_
from src.models2 import make_model
from args import make_args
import os.path as osp
from src.utils import load_checkpoint


def one_pass(model, cnf: CNFFormula, p, a=None):
    # @TODO cnf 
    adj_pos, adj_neg = cnf.graphs_polarity()
    xv = model(cnf.bipartite())   # assignment for variables
    sm = smooth_max_(xv, adj_pos, adj_neg, p, a)  # evaluation of clauses
    vp, vn = vars_of_clauses(sm, adj_pos), vars_of_clauses(sm, adj_neg)
    cp, cn = conflict_clauses(sm, vp, adj_pos), conflict_clauses(sm, vn, adj_neg)

    return torch.cat([vp, -vn]).tolist(), torch.unique(torch.cat([cp, cn]), sorted=True)


def conflict_select():
    return


def hybrid_cdcl(cnf_formula: CNFFormula,
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

    args = make_args()
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    model = make_model(args=args).to(device)
    last_epoch, loss = load_checkpoint(osp.join(args.save_root,
                                                args.save_name + '_' + str(args.last_epoch) + '.pickle'),
                                       model)

    # Unit propagation
    # propagated_literals, antecedent_of_conflict = cnf_formula.unit_propagation(decision_level)
    propagated_literals, antecedent_of_conflict = one_pass(cnf_formula)
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
        # propagated_literals, antecedent_of_conflict = cnf_formula.unit_propagation(decision_level)
        propagated_literals, antecedent_of_conflict = one_pass(cnf_formula, p=args.p)
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
            propagated_literals, antecedent_of_conflict = one_pass(model, cnf_formula, p=args.sm_par)
            unit_propagations += len(propagated_literals)

    return True, list(cnf_formula.assignment_stack), decisions, unit_propagations, restarts
