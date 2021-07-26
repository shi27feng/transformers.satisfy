import argparse
import sys


class Formula:
    """
    Class for representing NNF formula as a binary ordered tree and methods for transforming this formula to CNF.

        self.root - represents a node of the binary tree with logical connective or leaf variable

        self.left - left subformula which is again an instance of the class Formula, if self.root = "not" then the
        negated variable is in this subtree

        self.right - right subformula which is again an instance of the class Formula, if self.root = "not" then this
        subtree is None

        self.string_formula - original string formula represented as a list

        self.variable - fresh variable which is assigned to this `self.root` by `assign_fresh_variables`

        self.clauses - list of CNF clauses for this root, doesn't contain clauses of its subformulas

        self.fresh_variables_dict - dictionary representing a map from the original variables to the fresh variables
    """

    def __init__(self):
        self.root = None
        self.left = None
        self.right = None
        self.string_formula = None
        self.variable = None
        self.clauses = []
        self.fresh_variables_dict = {}

    def create_formula_from_string(self, string_formula):
        """
        Creates the formula in the form of a binary ordered tree from a given string in the input.

        :param string_formula: string formula representing NNF formula.
        """
        string_formula = string_formula.replace("(", "( ").replace(")", " )")
        string_formula = string_formula.split()
        self._recursive_formula_construction(string_formula)

    def _recursive_formula_construction(self, string_formula):
        """
        Recursively builds a binary ordered tree of the formula.

        :param string_formula: a list which elements are logical connectives, parentheses or variable names
                               representing a formula.
        """
        self.string_formula = string_formula

        while string_formula:
            e = string_formula[0]
            string_formula = string_formula[1:]

            if e == ")":
                return string_formula

            elif e in ["and", "or"]:
                self.root = e
                self.left = Formula()
                self.right = Formula()
                string_formula = self.left._recursive_formula_construction(string_formula)
                string_formula = self.right._recursive_formula_construction(string_formula)

            elif e == "not":
                self.root = e
                self.left = Formula()
                string_formula = self.left._recursive_formula_construction(string_formula)

            elif e != "(":
                self.root = e
                return string_formula

    def assign_fresh_variables(self, variable=1, fresh_variables_dict=None):
        """
        Assigns the fresh variables to the nodes in the derivation tree of the formula.

        :param variable: indicates the the value of the first fresh variable which is going to be used. The next
                         variable will be given the value of `variable` + 1 etc.
        :param fresh_variables_dict: dictionary to remember the map from the original variables in leaves to the fresh
                                     ones
        :return: variable, fresh_variables_dict: variable of the root, which is also the maximum value used, and the
                                                 dictionary to remember the map from original variables in leaves to the
                                                 fresh ones.
        """
        if fresh_variables_dict is None:
            fresh_variables_dict = {}

        if self.root:
            if self.left:
                variable, fresh_variables_dict = self.left.assign_fresh_variables(variable, fresh_variables_dict)

            if self.right:
                variable, fresh_variables_dict = self.right.assign_fresh_variables(variable, fresh_variables_dict)

            # If the self.root is a leaf of the formula, i.e. original variable, then check whether it has already been
            # assigned to a fresh variable. If it has been then use this variable, otherwise assign a new unused
            # fresh variable to it and remember this assignment in fresh_variables_dict.
            if self.root not in ["and", "or", "not"]:
                if self.root in fresh_variables_dict:
                    self.variable = fresh_variables_dict[self.root]

                else:
                    self.variable = variable
                    fresh_variables_dict.update({self.root: self.variable})
                    variable += 1

            # For `not` connective assign its fresh variable to the negative variable of its left child.
            elif self.root == "not":
                self.variable = -self.left.variable

            # Every `and` and `or` connective gets a new variable.
            else:
                self.variable = variable
                variable += 1

            self.fresh_variables_dict = fresh_variables_dict
            return variable, fresh_variables_dict

    def make_clauses(self, left_to_right):
        """
        Adds definition clauses for every node in the derivation tree into their variable `self.clauses`.

        :param left_to_right: Specifies if the Tseitin encoding should use equivalences (False) or only left-to-right
        implications (True).
        """

        if self.root == "and":
            if left_to_right:
                self.clauses.append([-self.variable, self.left.variable, 0])
                self.clauses.append([-self.variable, self.right.variable, 0])

            else:
                self.clauses.append([-self.variable, self.left.variable, 0])
                self.clauses.append([-self.variable, self.right.variable, 0])
                self.clauses.append([self.variable, -self.left.variable, -self.right.variable, 0])

            self.left.make_clauses(left_to_right)
            self.right.make_clauses(left_to_right)

        elif self.root == "or":
            if left_to_right:
                self.clauses.append([-self.variable, self.left.variable, self.right.variable, 0])

            else:
                self.clauses.append([-self.variable, self.left.variable, self.right.variable, 0])
                self.clauses.append([self.variable, -self.left.variable, 0])
                self.clauses.append([self.variable, -self.right.variable, 0])

            self.left.make_clauses(left_to_right)
            self.right.make_clauses(left_to_right)

    def get_clauses_to_list(self):
        """
        Extracts all clauses from the derivation tree nodes.

        :return: list of all clauses.
        """
        clauses = []
        if self.root:
            for clause in self.clauses:
                clauses.append(clause)

            if self.left:
                clauses += self.left.get_clauses_to_list()

            if self.right:
                clauses += self.right.get_clauses_to_list()

        return clauses

    def print_tseitin_encoding(self, clauses, output_file):
        """
        Prints final CNF formula.

        :param clauses: list of clauses to be printed.
        :param output_file: output file or `None` in the case of printing to stdout.
        """
        output = open(output_file, mode="w") if output_file else sys.stdout

        print("c root node variable =", self.variable, file=output)

        for original_variable, fresh_variable in self.fresh_variables_dict.items():
            print("c ", original_variable, "=", fresh_variable, file=output)

        print("p cnf", self.variable, len(clauses), file=output)
        for clause in clauses:
            for c in clause:
                print(c, end=" ", file=output)
            print(file=output)

        if output is not sys.stdout:
            output.close()

    def tseitin_encoding(self, left_to_right, output_file):
        """
        Transforms this formula in NNF into a DIMACS CNF formula using Tseitin encoding and prints it.

        :param left_to_right: Specifies if the Tseitin encoding should use equivalences (False) or only left-to-right
                              implications (True).
        :param output_file: output file or `None` in the case of printing to stdout.
        """
        if self.root:
            self.assign_fresh_variables()
            # add the root clause of the formula
            self.clauses.append([self.variable, 0])
            # make the clauses for every internal node
            self.make_clauses(left_to_right)
            # get all clauses to list
            clauses = self.get_clauses_to_list()
            self.print_tseitin_encoding(clauses, output_file)

    def print_formula(self, level=0):
        """
        Prints basic information about this formula.

        :param level: starting level of the root, usually 0 or 1.
        """
        print("Level: ", level, "Value: ", self.root, "Variable: ", self.variable)
        if self.left:
            self.left.print_formula(level=level + 1)
        if self.right:
            self.right.print_formula(level=level + 1)


def formula2cnf(input_file=None, output_file=None, left_to_right=False):
    """
    Transforms a NNF formula from `input_file` using Tseitin encoding to equisatisfiable formula in CNF, which is
    written to the `output_file`.

    :param input_file: Input file which contains a description of a formula in NNF. If `None` then stdin is used.
    :param output_file: Output file which contains an equisatisfiable DIMACS CNF formula obtained by Tseitin encoding.
                        If `None` then stdout is used.
    :param left_to_right: Specify if the Tseitin encoding should use only left-to-right implications instead of
                          equivalences.
    """
    input = open(input_file, mode="r") if input_file else sys.stdin
    string_formula = input.read()

    if input is not sys.stdin:
        input.close()

    formula = Formula()
    formula.create_formula_from_string(string_formula)
    formula.tseitin_encoding(left_to_right=left_to_right, output_file=output_file)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str, help="Input file which contains a description of a "
                                                                "formula in NNF")
    parser.add_argument("--output", default=None, type=str, help="Output file which contains an equisatisfiable "
                                                                 "DIMACS CNF formula obtained by Tseitin encoding")
    parser.add_argument("--ltr", default=False, action="store_true", help="Specify if the Tseitin encoding should use "
                                                                          "only left-to-right implications instead of"
                                                                          " equivalences")
    args = parser.parse_args()
    formula2cnf(input_file=args.input, output_file=args.output, left_to_right=args.ltr)
