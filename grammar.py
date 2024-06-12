#!/usr/bin/env python3


import random
import collections
from dataclasses import dataclass
import yaml
import json
import os
from argparse import ArgumentParser
import copy
import sympy
from sympy import symbols
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve
from typing import Optional
from openai import OpenAI
import re
from tree import VariableDeclaration, QuestionDeclaration, Num, Const, Var, Add, Sub, Mult, Div, Compare, Frac, Square, Cube, Exponent, Remainder, Expr, Node, BinOp
import math
import unittest
from anthropic import Anthropic



client = OpenAI()

LETTERS = LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
           
TRIVIAL_FILTER_PROB = 0.90
           
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

"""
CONTEXT AND STANDARD CLASSES
"""
class Context:
    def __init__(self, expressions, min_number, max_number, max_depth, min_length, max_length, min_value, max_value, unknowns, num_unknowns, min_coeff, max_coeff, skip_count, number_type, math_concept, custom_final_answer):
        self.expression = choice(expressions)
        self.min_number = min_number
        self.max_number = max_number
        self.max_depth = max_depth
        self.curr_depth = 0
        self.min_length = min_length
        self.max_length = max_length
        self.letters = LETTERS[:]
        self.letters_used = []
        self.min_value = min_value
        self.max_value = max_value
        self.unknowns = unknowns
        self.num_unknowns = num_unknowns
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.math_concept = math_concept
        self.skip_count = skip_count
        self.number_type = number_type
        self.custom_final_answer = custom_final_answer
        # self.vars_and_vals = {}


class Standard:
    def __init__(self, id, description, short_description, filters, transforms, expressions,
                 min_number, max_number, max_depth, min_length, max_length, min_value, max_value, unknowns, num_unknowns, min_coeff, max_coeff, skip_count, number_type, custom_final_answer, types_of_fup, math_concept, samples):
        self.id = id
        self.description = description
        self.short_description = short_description
        self.filters = filters
        self.transforms = transforms
        self.expressions = expressions
        self.min_number = min_number
        self.max_number = max_number
        self.max_depth = max_depth
        self.min_length = min_length
        self.max_length = max_length
        self.min_value = min_value
        self.max_value = max_value
        self.unknowns = unknowns
        self.num_unknowns = num_unknowns
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.skip_count = skip_count
        self.number_type = number_type
        self.custom_final_answer = custom_final_answer
        self.types_of_fup = types_of_fup
        self.math_concept = math_concept
        self.samples = samples

    def __str__(self):
        return f'{self.id}'

    def grade(self) -> str:
        return self.id.split('.')[0]

    def get_context_for_standard(self) -> Context:
        return Context(self.expressions, self.min_number, self.max_number, self.max_depth, self.min_length, self.max_length, self.min_value, self.max_value, self.unknowns, self.num_unknowns, self.min_coeff, self.max_coeff, self.skip_count, self.number_type, self.custom_final_answer, self.math_concept)

# TODO: Problem should move to tree.py and be detached from Standard. Once we have a
# ProblemGenerationResult class, that one can contain the Standard that was used to
# generate the problem.
@dataclass
class Problem(Node):
    'Symbolic representation of a mathematical problem.'
    variables: list[VariableDeclaration]
    question: QuestionDeclaration
    standard: Standard
    
    def __str__(self) -> str:
        lines = [str(vd) for vd in self.variables]
        lines.append(str(self.question))
        # lines.append(str(self.standard))
        #
        return '\n'.join(lines)
    
    def __eq__(self, p2) -> bool:
        return str(self) == str(p2)

    @staticmethod
    def parse(s: str, standard: Standard = None) -> 'Problem':
        arr = s.split('\n')
        question = arr[-1]
        variables = arr[:-1]
        q_pattern = r'\[\[question\s+([^\s]+)\s*=\s*\[\'([^\]]+)\'\]\]\]'
        q_match = re.match(q_pattern, question)

        if q_match:
            q_x = q_match.group(1)
            q_y = q_match.group(2)
        else:
            raise ValueError(f'Last statement is not a valid question: {question}')

        q_dec = QuestionDeclaration(q_x, [q_y])
        v_pattern = r'\[\[var\s+([^\s]+)\s*=\s*([^\]]+)\]\]'

        problem_vars = []
        for var in variables:
            v_match = re.match(v_pattern, var)
            if v_match:
                v_x = v_match.group(1)
                v_y = v_match.group(2)
            else:
                return False
            v_dec = VariableDeclaration(v_x, v_y)
            problem_vars.append(v_dec)
        return Problem(problem_vars, q_dec, standard)
    
    def subtree_nodes(self):
        subtree_nodes = []
        for var in self.variables:
            subtree_nodes.extend(var.subtree_nodes())
        subtree_nodes.extend(self.question.subtree_nodes())
        return subtree_nodes
        
    def replace_node(self, index, newNode, qOnly):
        # track curr starting with curr = 0
        # update: curr += len(current var) for var in self.variables - this is to check which statement we are updating the number in
        # go into that statement, call replace_node() on that vardec with the edited index (index - curr) and newNode
        new_p = copy.deepcopy(self)
        if not qOnly:
            count = 0
            var_to_change = new_p.variables[0]
            for var in new_p.variables:
                curr_subtree_nodes = var.subtree_nodes()
                count += len(curr_subtree_nodes)
                if index <= count:
                    var_to_change = var
                    count -= len(curr_subtree_nodes)
                    break
            var_to_change.replace_node(index - count, newNode)
        else:
            new_p.question.replace_node(index, newNode)
        return new_p
        
        
    """
    def children(self) -> list[Node]:
        return self.variables + [self.question]

    def set_child(self, index, node):
        if 0 <= index < len(self.variables):
            assert isinstance(node, VariableDeclaration)
            self.variables[index] = node
        elif index == len(self.variables):
            assert isinstance(node, QuestionDeclaration)
            self.question = node
        raise ValueError('Invalid child index.')
    """

@dataclass
class GeneratedProblem:
    problem: 'Problem'
    followup_problems: Optional[list['Problem']] = None
    theme: Optional[str] = None

    def to_json(self):
        return json.dumps({
            "problem": self.problem,
            "followup_problems": self.followup_problems,
            "theme": self.theme
        })
    
    def __str__(self) -> str:
        to_print = str(self.problem)
        to_print += [str(p) for p in self.followup_problems]
        if self.theme != None:
            to_print += '\n' + self.theme
        return to_print
    

"""
PROBLEM FILTER CLASSES
"""

class ProblemFilter:
    name: str
    
    def satisfies(self, p: Problem) -> bool:
        raise NotImplementedError
        
class ProblemTransformation:
    name: str
    
    def transform(self, p: Problem) -> Problem:
        raise NotImplementedError

class CheckIntermediateValues(ProblemFilter):
    def __init__(self, min_value: int, max_value: int):
        self._min_value = min_value
        self._max_value = max_value
        self.name = f"Minimum Intermediate Value = {self._min_value}, Maximum Intermediate Value = {self._max_value}"
    
    
    def satisfies(self, p: Problem) -> bool:
        context = p.standard.get_context_for_standard()
        # splitting into two cases because sympy runs as fast for the unknowns case as eval does for the regular case
        if not context.unknowns:
            q = p.question
            vars = p.variables
            g = {}
            for var in vars:
                g[var.name] = eval(str(var.value), g)
                if(g[var.name] > self._max_value or g[var.name] < self._min_value):
                    return False
                if len(p.variables) == 1 and isinstance(var.value, Num):
                    return False
                expr_check = check_expression(context, var.value)
                if not expr_check:
                    return False
            for var in q.question:
                expr_check = check_expression(context, var)
                if not expr_check:
                    return False
            # temporary fix???
            final_ans = calculate_final_answer(p)
            if final_ans > self._max_value or final_ans < self._min_value:
                return False
            # print("satisfies check intermediate values")
            return True
        else:
            # print("problem is: ", str(p))
            for var in p.variables:
                if isinstance(var.name, Num):
                    if var.name.value > self._max_value or var.name.value < self._min_value:
                        return False
            intermediate_vals = calculate_intermediate_answers(p)
            # print(intermediate_vals)
            if intermediate_vals == None:
                return False
            # print(intermediate_vals)
            for val in intermediate_vals.values():
                # print(val)
                if val > self._max_value:
                    return False
                if val < self._min_value:
                    return False
            # print("satisfies check intermediate values")
            return True

    def __str__(self):
        return self.name
    
class ChainsOfVariables(ProblemFilter):
    def __init__(self):
        self.name = f"Chains Of Variables"
        
    def satisfies(self, p: Problem) -> bool:
        q = p.question
        vars = p.variables
        for var in vars:
            if isinstance(var.value, Var):
                return False
        # print("satisfies chains of vars")
        return True

    def __str__(self):
        return self.name

class ProblemLength(ProblemFilter):
    def __init__(self, min_length, max_length):
        self._min_length = min_length
        self._max_length = max_length
        self.name = f"Problem Length Filter. Minimum Length = {self._min_length}, Maximum Length = {self._max_length}"
    
    def satisfies(self, p: Problem) -> bool:
        if len(p.variables) == 1:
            if isinstance(p.variables[0].value, Num) or isinstance(p.variables[0].value, Frac):
                return False
        if self._min_length <= len(p.variables) <= self._max_length:
            # print("satisfies problem length")
            return True
        return False

class ContainsTen(ProblemFilter):
    def __init__(self):
        pass

    def satisfies(self, p: Problem) -> bool:
        for var in p.variables:
            if var.name != Num(10):
                for child in var.subtree_nodes():
                    if p.standard.id == "K.OA.A.4" or p.standard.id == "K.NBT.A.1":
                        if child == Num(10):
                            return True
            else:
                return True
        return False



class NoFilter(ProblemFilter):
    def __init__(self):
        pass

    def satisfies(selfself, p: Problem) -> bool:
        return True
"""
GENERAL FUNCTIONS
"""
def calculate_intermediate_answers(problem: Problem) -> str:
    def create_eq(a, b):
        return sympy.Eq(parse_expr(a), parse_expr(b))
    equations = [create_eq(str(v.name), str(v.value)) for v in problem.variables]
    if problem.standard.num_unknowns < 2:
        equations.append(create_eq(str(problem.question.name), str(problem.question.question[0])))
    """
    else:
        equations.append(create_eq(str(problem.question.name), str(tuple(problem.question.question))))
    """
    sympy_equations = [sympy.simplify(eq) for eq in equations]
    # print(sympy_equations)
    solutions = solve(sympy_equations)
    if isinstance(solutions, list):
        if solutions == []:
            return None
        else:
            solutions = solutions[0]
    return solutions

def calculate_final_answer(problem: Problem) -> str:
    if problem.standard.custom_final_answer == "None":
        ans = calculate_intermediate_answers(problem)
        if ans != None:
            if problem.standard.num_unknowns < 2:
                final_answer = ans[sympy.symbols(problem.question.name)]
            else:
                final_answer = ans
            return final_answer
        else:
            return None
    elif problem.standard.custom_final_answer == "Factor Pairs":
        factor_pairs = []
        to_factor = problem.question.question[0].value
        for i in range(1, int(math.sqrt(to_factor)) + 1):
            if to_factor % i == 0:
                factor_pairs.append((i, int(to_factor/i)))
        return factor_pairs
    elif problem.standard.custom_final_answer == "System of Equations":
        return calculate_intermediate_answers(problem)
    elif problem.standard.custom_final_answer == "Compare":
        question = problem.question.question[0]
        if eval(str(question.lhs)) > eval(str(question.rhs)):
            return ">"
        elif eval(str(question.lhs)) < eval(str(question.rhs)):
            return "<"
        else:
            return "="


    
def make_question(context: Context) -> QuestionDeclaration:
    v = get_var(context)
    e = context.expression(context)
    dec = QuestionDeclaration(v, [e])
    # context.vars_and_vals[v] = e
    return dec
    
def make_question_with_unknowns(context: Context) -> QuestionDeclaration:
    v = get_var(context)
    e = context.letters_used
    dec = QuestionDeclaration(v, e)
    # context.vars_and_vals[v] = e
    return dec
    
def get_var(context: Context):
    to_ret = random.choice(context.letters)
    context.letters.remove(to_ret)
    # cannot append to letters_used here because then variables used can show up in their own expression
    return to_ret
        
def choice(rules: list):
    def f(context: Context):
        r = random.choice(rules)
        return r(context)
    return f
     

def get_used_variables(expr: Expr) -> set[str]:
    if isinstance(expr, Num) or isinstance(expr, Const):
        return set()
    if isinstance(expr, int):
        return set()
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, str):
        return {expr}
    if isinstance(expr, BinOp):
        return get_used_variables(expr.lhs).union(get_used_variables(expr.rhs))
    assert False, "Uncovered cases"
    
def refine(p: Problem) -> Problem:
    for t in p.standard.transforms:
        p = t.transform(p)
    return p

def check_validity_for_problem_addition(p: Problem, context: Context) -> bool:
    # look up which checks apply to the standard
    # check for useless variables
    # check max constant
    # check that a certain operation was used
    for filter in p.standard.filters:
        if "Problem Length" not in filter.name:
            if not filter.satisfies(p):
                return False
    return True
    
def check_validity(p: Problem, context: Context) -> bool:
    # look up which checks apply to the standard
    # check for useless variables
    # check max constant
    # check that a certain operation was used
    for filter in p.standard.filters:
        if not filter.satisfies(p):
            return False
    return True


def modify_problem(p: Problem, context: Context) -> Problem:
    # print("old problem", p)
    new_valid_prob = False
    count = 0
    while not new_valid_prob:
        if count > 1000:
            return None
        subtree_nodes = p.subtree_nodes()
        # print("subtree nodes", subtree_nodes)
        index = random.randint(0, len(subtree_nodes) - 1)
        checkNum = False
        while not checkNum:
            if isinstance(subtree_nodes[index], Num):
                checkNum = True
                break
            else:
                index = random.randint(0, len(subtree_nodes) - 1)
        # print(f"index {index}")
        new_num = number(context, True)
        # print("new num", new_num)
        new_p = copy.deepcopy(p)
        # the third argument passes true if the structure consists of only a question
        new_p = new_p.replace_node(index, new_num, len(p.variables) == 0)
        new_p = refine(new_p)
        count += 1
        if check_validity(new_p, context) and new_num != subtree_nodes[index]:
            new_valid_prob = True
        # print("new problem", new_p)
    
    return new_p
    
def add_to_problem(p: Problem, context: Context) -> Problem:
    new_valid_prob = False
    count = 0
    while not new_valid_prob:
        if count > 1000:
            return None
        new_p = copy.deepcopy(p)
        new_context = copy.deepcopy(context)
        new_p.variables.append(make_stmt(new_context))
        new_p.question = make_question(new_context)
        new_p = refine(new_p)
        count += 1
        # no way to use check_validity on new_p, because the new problem
        # by def. check_validity_for_problem_addition checks with all filters
        # except the problem length filter. the only issue now is that there
        # still seem to be some problems that this function simply cannot
        # generate follow ups for, no matter how long this while loop loops for
        # if you remove the "check_validity_for_problem_addition(new_p, new_context)",
        # this code works better, but then there is no guarantee of the final answer being in the right range
        if new_p != p and len(new_p.variables) > len(p.variables) and check_validity_for_problem_addition(new_p, new_context):
            new_valid_prob = True
    return new_p



    new_valid_prob = False
    while not new_valid_prob:
        subtree_nodes = p.subtree_nodes()
        # print(subtree_nodes)
        index = random.randint(0, len(subtree_nodes) - 1)
        checkNum = False
        while not checkNum:
            if isinstance(subtree_nodes[index], Num):
                checkNum = True
                break
            else:
                index = random.randint(0, len(subtree_nodes) - 1)
        # print(f"index {index}, subtree_nodes {subtree_nodes[index]}")
        new_num = number(context)
        new_p = copy.deepcopy(p)
        new_p = new_p.replace_node(index, new_num)
        new_p = refine(new_p)
        if check_validity(new_p, context) and new_num != subtree_nodes[index]:
            new_valid_prob = True
    
    return new_p
    

    
def generate_many(n: int, standard: Standard, n_fup: int = None) -> list[GeneratedProblem]:
    problems = []

    while len(problems) < n:
        p = None
        context = standard.get_context_for_standard()
        if not standard.unknowns:
            length = random.randint(context.min_length, context.max_length)
            # 2*length was here
            p = generate_problem(context, standard, length)
            p = refine(p)
            while len(p.variables) < length or not check_validity(p, context):
                context = standard.get_context_for_standard()
                # 2*length was here
                p = generate_problem(context, standard, length)
                p = refine(p)
        else:
            p = generate_problem_with_unknowns(context, standard)
        if p is not None and check_validity(p, context):
            gp = GeneratedProblem(p)
            if n_fup is not None:
                gp.followup_problems = []
                # fup_p can be set to modify_problem or add_to_problem
                if 1 in standard.types_of_fup:
                    print("generating first follow up")
                    if not (p.standard.id == "1.OA.A.2" and calculate_final_answer(p) == p.standard.max_value):
                        new_problem = add_to_problem(p, context)
                        if new_problem != None:
                            gp.followup_problems.append((1, new_problem))
                if 2 in standard.types_of_fup:
                    print("generating second follow up")
                    new_problem = modify_problem(p, context)
                    if new_problem != None:
                        gp.followup_problems.append((2, new_problem))
            problems.append(gp)
            print(f"{len(problems)} symbolic structures of {n} generated.")

    return problems

def make_stmt(context: Context) -> VariableDeclaration:
    v = get_var(context)
    context.curr_depth = 0
    e = context.expression(context)
    context.letters_used.append(v)
    dec = VariableDeclaration(v, e)
    return dec

def min_max_bound(context: Context, value):
    if isinstance(value, Num) or isinstance(value, Frac):
        value = value.value # this looks very funky, but handles the edge case where fractions are Num/Num, so fraction.lhs.value = Num, and comparing a Num with an int causes issues
    return not (value > context.max_value or value < context.min_value)

def check_expression(context: Context, expression: Expr) -> bool:
    if isinstance(expression, BinOp):
        if isinstance(expression.lhs, Num):
            if not min_max_bound(context, expression.lhs.value):
                return False
        else:
            if not check_expression(context, expression.lhs):
                return False
        if isinstance(expression.rhs, Num):
            if not min_max_bound(context, expression.rhs.value):
                return False
        else:
            if not check_expression(context, expression.rhs):
                return False
    else:
        if isinstance(expression, Num):
            if not min_max_bound(context, expression.value):
                return False
    return True
    
def generate_problem(context: Context, standard: Standard, length: int) -> Problem:
    """
    x = random.randint(context.min_length, context.max_length)
    print("problem length:", x)
    """
    p = Problem([], None, standard)
    for _ in range(length):
        p.variables.append(make_stmt(context))
        # context.question_opts = choice(context.choices)
    p.question = make_question(context)
    # print("problem: ", p)
    return p
        
def generate_problem_with_unknowns(context: Context, standard: Standard) -> Expr:
    # print("-----------------------------------")
    # in a problem with unknowns, the min_length MUST equal max_length MUST equal num_unknowns
    first_expr = context.expression(context)
    first_expr_ans = Num(int(eval(str(first_expr))))
    count = 0
    # print("first expression", first_expr)
    while count < context.num_unknowns:
        context.curr_depth = 0
        subtree_nodes = first_expr.subtree_nodes()
        index = random.randint(0, len(subtree_nodes) - 1)
        checkNum = False
        while not checkNum:
            if isinstance(subtree_nodes[index], Num):
                checkNum = True
                break
            else:
                index = random.randint(0, len(subtree_nodes) - 1)
        new_value = Var(get_var(context))
        coeff = random.randint(context.min_coeff, context.max_coeff)
        context.letters_used.append(new_value)
        if coeff != 1:
            new_value = Mult(Const(coeff), new_value)
            # print("new value", new_value)
        new_first_expr = copy.deepcopy(first_expr)
        # print("subtree nodes", subtree_nodes)
        # print("index to replace at", index)
        new_first_expr.replace_to_var(index, new_value)
        # print("new first expr", new_first_expr)
        count += 1
        first_expr = new_first_expr
    problem = Problem([], None, standard)
    problem.variables.append(VariableDeclaration(first_expr_ans, new_first_expr))
    for j in range(1, context.num_unknowns):
        context.curr_depth = 0
        curr_expr = context.expression(context)
        curr_expr_ans = Num(int(eval(str(curr_expr))))
        count = 0
        while count < context.num_unknowns:
            subtree_nodes = curr_expr.subtree_nodes()
            index = random.randint(0, len(subtree_nodes) - 1)
            checkNum = False
            while not checkNum:
                if isinstance(subtree_nodes[index], Num):
                    checkNum = True
                    break
                else:
                    index = random.randint(0, len(subtree_nodes) - 1)
            new_value = context.letters_used[count]
            coeff = random.randint(context.min_coeff, context.max_coeff)
            if coeff != 1:
                new_value = Mult(Const(coeff), new_value)
            new_curr_expr = copy.deepcopy(curr_expr)
            new_curr_expr.replace_to_var(index, new_value)
            curr_expr = new_curr_expr
            count += 1
        problem.variables.append(VariableDeclaration(curr_expr_ans, new_curr_expr))
    problem.question = make_question_with_unknowns(context)
    # print("---")
    # print(problem)
    return problem
    
    
    """
    first_expr = context.expression(context)
    first_expr_ans = Num(int(eval(str(first_expr))))
    count = 0
    while count < context.num_unknowns:
        children = first_expr.subtree_nodes()
        expr_i = random.randint(0, len(children) - 1)
        expr = children[expr_i]
        while not isinstance(expr, Num):
            expr_i = random.randint(0, len(children) - 1)
            expr = children[expr_i]
        new_value = Var(get_var(context))
        coeff = random.randint(context.min_coeff, context.max_coeff)
        context.letters_used.append(new_value)
        if coeff != 1:
            new_value = Mult(Const(coeff), new_value)
        new_first_expr = copy.deepcopy(first_expr)
        new_first_expr.set_child(expr_i, new_value)
        count += 1
        first_expr = new_first_expr
    problem = Problem([], None, standard)
    problem.variables.append(VariableDeclaration(first_expr_ans, new_first_expr))
    for j in range(1, context.num_unknowns):
        context.curr_depth = 0
        curr_expr = context.expression(context)
        curr_expr_ans = Num(int(eval(str(curr_expr))))
        count = 0
        while count < context.num_unknowns:
            children = curr_expr.children()
            expr_i = random.randint(0, len(children) - 1)
            expr = children[expr_i]
            while not isinstance(expr, Num):
                expr_i = random.randint(0, len(children) - 1)
                expr = children[expr_i]
            new_value = context.letters_used[count]
            coeff = random.randint(context.min_coeff, context.max_coeff)
            if coeff != 1:
                new_value = Mult(Const(coeff), new_value)
            new_curr_expr = copy.deepcopy(curr_expr)
            new_curr_expr.set_child(expr_i, new_value)
            curr_expr = new_curr_expr
            count += 1

        problem.variables.append(VariableDeclaration(curr_expr_ans, new_curr_expr))
    problem.question = make_question_with_unknowns(context)
    # print("problem is:", problem)
    return problem
    """


def parse_expression_list(l: list[str]) -> list:
    fns = []
    for fname in l:
        fns.append(
            {
                "variable": variable,
                "number": number,
                "count":count,
                "addition": addition,
                "subtraction": subtraction,
                "multiplication": multiplication,
                "division": division,
                "remainder": remainder,
                "mult_4nbtb5": mult_4nbtb5,
                "compare": compare,
                "triangle_perimeter": triangle_perimeter,
                "triangle_area": triangle_area,
                "quadrilateral_perimeter": quadrilateral_perimeter,
                "rectangle_perimeter": rectangle_perimeter,
                "rectangle_area": rectangle_area,
                "polygon_perimeter": polygon_perimeter,
                "div_5nbtb6": div_5nbtb6,
                "div_5nbtb7": div_5nbtb7,
                "square": square,
                "cube": cube,
                "exponent": exponent
            }[fname]
        )
    return fns


def parse_transform_list(l: list[str]) -> list:
    fns = []
    for fname in l:
        fns.append(
            {
                "NoUselessVariables": NoUselessVariables(),
                "Simplify": Simplify(),
                "NoTransform": NoTransform()
            }[fname]
        )
    return fns


def parse_filter_list(l: list[str], min_value, max_value, min_length, max_length) -> list:
    fns = []
    for fname in l:
        fns.append(
            {
                "CheckIntermediateValues": CheckIntermediateValues(min_value, max_value),
                "ChainsOfVariables": ChainsOfVariables(),
                "ProblemLength": ProblemLength(min_length, max_length),
                "ContainsTen": ContainsTen(),
                "NoFilter": NoFilter()
            }[fname]
        )
    return fns


def parse_custom_final_answer_list(l:list[str]) -> list:
    fns = []
    for fname in l:
        fns.append(
            {
                "CheckIntermediateValues": CheckIntermediateValues(min_value, max_value),
                "ChainsOfVariables": ChainsOfVariables(),
                "ProblemLength": ProblemLength(min_length, max_length),
                "ContainsTen": ContainsTen(),
                "NoFilter": NoFilter()
            }[fname]
        )
    return fns

def load_standards() -> list[Standard]:
    with open('commoncore.yaml') as f:
        objs = yaml.safe_load(f)
        
    standards = {}
        
    for o in objs:
        expressions = parse_expression_list(o["expressions"])
        filters = parse_filter_list(o["filters"], o["min_value"], o["max_value"], o["min_length"], o["max_length"])
        transforms = parse_transform_list(o["transforms"])
        standards[o["id"]] = Standard(
            o["id"],
            o["description"],
            o["short_description"],
            filters, transforms, expressions,
            o["min_number"],
            o["max_number"],
            o["max_depth"],
            o["min_length"],
            o["max_length"],
            o["min_value"],
            o["max_value"],
            o["unknowns"],
            o["num_unknowns"],
            o["min_coeff"],
            o["max_coeff"],
            o["skip_count"],
            o["number_type"],
            o["custom_final_answer"],
            o["types_of_fup"],
            o["math_concept"],
            o["samples"]
        )

    return standards

def generate_all_sym_structs(standards: list[Standard], n, n_fup):
    for standard in standards:
        print(standard)
        print(standards[standard].description)
        g_problems = generate_many(n, standards[standard], n_fup)
        for i, g_problem in enumerate(g_problems):
            print(f"Problem #{i}")
            p_seq = [g_problem.problem]
            for problem in p_seq:
                print(problem)
                
def generate_theme(problem: Problem):
    objects_list = ["Ball", "Dog", "Cat", "Car", "Bike", "Tree", "House", "Sun", "Moon", "Star","Baby", "Friend",
                    "Book", "Toy", "Chair", "Table","Food", "Water", "Juice", "Milk", "Apple", "Banana", "Pizza",
                    "Cake","Ice cream", "Candy", "Cookie", "Chocolate", "Cheese", "Sandwich", "Chicken","Shark",
                    "Carrot", "Potato", "Tomato", "Cucumber", "Strawberry", "Grapes", "Lemon", "Orange","Train",
                    "Bus", "Truck", "Boat", "Plane", "Helicopter", "Doll", "Teddy bear","Puzzle", "Drum", "Guitar",
                    "Piano", "TV", "Phone", "Computer", "Bed","Pillow", "Blanket", "Lamp", "Clock", "Window", "Door",
                    "Toothbrush", "Toothpaste", "Soap", "Towel", "Shirt", "Pants", "Shoes", "Hat", "Jacket", "Sock",
                    "Dress", "Skirt", "Ballerina", "Superhero", "Princess", "Pirate", "Doctor", "Teacher",
                    "Firefighter","Police", "Farmer", "Zoo", "Park", "Beach", "Forest", "River", "Mountain","Farm",
                    "Circus", "Rain", "Snow", "Cloud", "Rainbow", "Wind","Flower", "Butterfly", "Bee", "Frog", "Fish",
                    "Turtle", "Elephant", "Giraffe", "Lion","Tiger", "Bear", "Monkey", "Penguin", "Dolphin", "Starfish",
                    "Rocket", "Astronaut", "Alien","Robot", "UFO", "Castle", "Pirate ship", "Treasure", "Fairy",
                    "Dragon", "Wizard", "Mermaid","Unicorn", "Dinosaur", "Bubble", "Slide", "Swing", "Sandbox",
                    "Jungle gym", "Beach ball", "Kite", "Scooter", "Skateboard", "Roller skates","Ice skates",
                    "Snowman", "Snowball", "Hot chocolate", "Marshmallow", "Fireplace", "Campfire", "Tent", "Gift",
                    "Sleeping bag", "Treasure chest", "Telescope", "Binoculars", "Backpack", "Map", "Compass", "Wallet",
                    "Camera", "Keys", "Sunglasses", "Watch", "Mouse", "Money", "Ring", "Necklace", "Bracelet",
                    "Earrings", "Glasses", "Balloon", "Candle", "Party", "Card", "Game", "Color", "Shape", "Number",
                    "Letter", "Word", "Song", "Toy car", "Stuffed animal", "Building blocks","Crayon", "Pen", "Pencil",
                    "Eraser", "Paint", "Colored pencil", "Cow", "School bus", "Bottle", "Snail", "Rope"]
    if problem.standard.math_concept == "None":
        s = "theme: "
        return s + str(random.choice(objects_list))
    else:
        s = "theme: "
        m = f"math concept: {problem.standard.math_concept}"
        # return s + str(random.choice(objects_list)) + "\n"+ m
        return m


def generate_problems(standard_id: str, n: int, output_path: str):
    if os.path.exists(output_path):
        with open(output_path) as f:
            problems = json.load(f)
    else:
        problems = []

    # Generate n more problems, store in problems
    # { "id": ..., "standard": "...", "statement": "...", "answer": "..." }
    with open(output_path, 'w') as f:
        json.dump(problems, f)


def check_sym_struct(s, final_ans, standard):
    try:
        problem = Problem.parse(s, standard)
        ans = calculate_final_answer(problem)
        return ans == final_ans
    except Exception as e:
        print("Symbolic structure error:", e)
        return False

    
def initialize_json(json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump([], f)
        
def write_to_json(data, json_file_path):
    # Writing to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def append_to_json(data, json_file_path):
    # Writing to the JSON file
    with open(json_file_path, 'a') as f:
        json.dump(data, f, indent=2)


"""
OPERATION FUNCTIONS
"""
def count(context: Context) -> Expr:
    # THIS FUNCTION DOES NOT WORK
    start = number(context)
    # end = number(context)
    skip = random.choice(context.skip_count)
    p = Problem([], None, standard)
    p.variables.append(VariableDeclaration(Var("start"), start))
    p.variables.append(VariableDeclaration(Var("skip"), skip))
    p.question = QuestionDeclaration(Var("next"), get_var(context))
    return p


def addition(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        return Add(context.expression(context), context.expression(context))
    else:
        return number(context)
        
def subtraction(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        return Sub(context.expression(context), context.expression(context))
    else:
        return number(context)
    
def multiplication(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        return Mult(context.expression(context), context.expression(context))
    else:
        return number(context)
        
def division(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        if "Fraction" in context.number_type:
            divisor = number(context)
            dividend = number(context)
            return Div(Num(divisor), Num(dividend))
        divisor = number(context)
        while(divisor.value == 0): # avoids divide by zero error
            divisor = number(context)
        quotient = number(context)
        dividend = eval(f"{divisor.value} * {quotient.value}")
        dividend = Num(dividend)
        return Div(dividend, divisor)
    else:
        return  number(context)

def remainder(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        a = number(context)
        b = number(context)
        if a.value < b.value:
            a, b, = b, a
        return Remainder(a, b)
    else:
        return number(context)

def mult_4nbtb5(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        rand = random.randint(0, 1)
        if rand == 0:
            first = Num(random.randint(10, 99))
            second = Num(random.randint(10, 99))
            return Mult(first, second)
        else:
            first = Num(random.randint(0, 9))
            second = Num(random.randint(1000, 9999))
            rand2 = random.randint(0, 1)
            if rand2 == 0:
                return Mult(first, second)
            else:
                return Mult(second, first)
    else:
        return Var(context.letters_used[0])

def compare(context:Context) -> Expr:
    lhs = number(context)
    rhs = number(context)
    return Compare(lhs, rhs)



def triangle_perimeter(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 1
        regular = random.randint(0, 1)
        if regular:
            return Mult(Const(3), number(context))
        else:
            a = number(context)
            b = number(context)
            c = number(context)
            return Add(Add(a, b), c)
        """
        # triangle
        if shape == 3:
            # equal sides
            if regular:
                s = random.randint(context.min_number, context.max_number)
                return Mult(3, s)
            # isoceles or scalene
            else:
                a = random.randint(context.min_number, context.max_number)
                b = random.randint(context.min_number, context.max_number)
                
            
        elif shape == 4:
        
        else:
        
        """
    return Var(context.letters_used[0])
    
def quadrilateral_perimeter(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 1
        regular = random.randint(0, 2)
        if regular==0:
            return Mult(Const(4), number(context))
        elif regular==1:
            a = number(context)
            b = number(context)
            c = number(context)
            d = number(context)
            return Add(Add(a, b), Add(c, d))
        else:
            context.curr_depth -= 1 # reset so that the rectangle_periemter function does not automatically try to return a variable
            return rectangle_perimeter(context)
    return Var(context.letters_used[0])

def rectangle_perimeter(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        num_one = number(context)
        num_two = number(context)
        return Add(Mult(Const(2), num_one), Mult(Const(2), num_two))
    return Var(context.letters_used[0])

    
def polygon_perimeter(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 1
        sides = random.randint(5, 11)
        irregular = random.randint(0, 1)
        if irregular == 1:
            vals = []
            for _ in range(sides):
                vals.append(number(context))
            a = Add(vals[0], vals[1])
            for val in vals[2:]:
                a = Add(a, val)
            return a
        else:
            return Mult(Const(sides), number(context))
    return Var(context.letters_used[0])
    
def triangle_area(context: Context) -> Expr:
    """
    context.curr_depth += 1
    half = Num(0.5)
    base = number(context)
    height = number(context)
    return Mult(Mult(half, base), height)
    """
    print("made it to area", context.curr_depth, context.max_depth)
    if context.curr_depth < context.max_depth:
        print("made it inside if", context.curr_depth, context.max_depth)
        context.curr_depth += 1
        half = Const(0.5)
        base = number(context)
        height = number(context)
        a = Mult(Mult(half, base), height)
        print(a)
        return a
    return Var(context.letters_used[0])

def rectangle_area(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        width = number(context)
        length = number(context)
        return Mult(width, length)
    return Var(context.letters_used[0])


def div_5nbtb6(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        divisor = Num(random.randint(1, 99))
        quotient = Num(random.randint(1, 99))
        dividend = divisor.value * quotient.value
        dividend = Num(dividend)
        return Div(dividend, divisor)
    else:
        return number(context)

def div_5nbtb7(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        divisor = Num(round(random.random()*context.max_number, 1))
        quotient = Num(round(random.random()*context.max_number, 1))
        dividend = round(divisor.value * quotient.value, 2)
        dividend = Num(dividend)
        return Div(dividend, divisor)
    else:
        return number(context)

# replaceFrac checks if the number we are generating is to replace a num/denom in a fraction
def number(context: Context, replaceFrac=False) -> Expr:
    context.curr_depth += 1
    type = random.choice(context.number_type)
    if type == "Int" or replaceFrac:
        x = Num(random.randint(context.min_number,
                           context.max_number))
    elif type == "Decimal-2":
        x = Num(round(random.random()*context.max_number, random.randint(1, 2)))
    elif type == "Decimal-3":
        x = Num(round(random.random() * context.max_number, random.randint(1, 3)))
    elif type == "Fraction":
        numerator = random.randint(context.min_number, context.max_number)
        denominator = random.randint(context.min_number, context.max_number)
        while(denominator == 0): # avoids divide by zero error
            denominator = random.randint(context.min_number, context.max_number)
        # x = f"{numerator} / {denominator}"
        x = Frac(Num(numerator), Num(denominator))
    return x

def variable(context: Context) -> Expr:
    context.curr_depth += 1
    if context.letters_used == []:
        return number(context)
    else:
        return Var(random.choice(context.letters_used))


def square(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        return Square(number(context))
    return number(context)


def cube(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        return Cube(number(context))
    return number(context)

def exponent(context: Context) -> Expr:
    if context.curr_depth < context.max_depth:
        context.curr_depth += 2
        lhs = number(context)
        rhs = number(context)
        return Exponent(lhs, rhs)
    return number(context)
"""
TRANSFORMS
"""
class NoUselessVariables(ProblemTransformation):
    def __init__(self):
        self.name = "No Useless Variables"

    def transform(self, p: Problem) -> Problem:
        # dependencies[v] = the set of variables that v depends on
        # print(p)
        dependencies = collections.defaultdict(set)
        for v in p.variables:
            # print(v)
            used_in_v = get_used_variables(v.value)
            dependencies[v.name] = set()
            for dep in used_in_v:
                dependencies[v.name].add(dep)
                dependencies[v.name].update(dependencies[dep])
        q = p.question
        used_in_q = set()
        for var in q.question:
            used_in_q.update(get_used_variables(var))
        dependencies[q.name] = set()
        for dep in used_in_q:
            dependencies[q.name].add(dep)
            dependencies[q.name].update(dependencies[dep])
        p.variables = [v for v in p.variables
                       if v.name in dependencies[q.name]]
        return p

def _is_numeric_constant(s: str) -> bool:
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?(/\d+)?$|^([a-zA-Z_]\w*)$'
    # pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?(/\d+)?$'
    return bool(re.match(pattern, s))


class Simplify(ProblemTransformation):
    def __init__(self):
        self.name = "Simplify"

    def transform(self, p: Problem) -> Problem:
        def replace_constants(expr):
            if str(expr) in propagated_constants:
                return replace_constants(var_to_val[str(expr)])
            if str(expr) in var_to_var:
                return replace_constants(var_to_var[str(expr)])
            if isinstance(expr, BinOp):
                expr_subst = copy.copy(expr)
                expr_subst.lhs = replace_constants(expr.lhs)
                expr_subst.rhs = replace_constants(expr.rhs)
                return expr_subst
            return expr
            
        # if a variable = number, and if said variable is used only once, replace it with the number
        # loop through variable declarations in the problem
        # check if the statement is of form "var = num"
        # if yes, add that var to dictionary "var_freq[var] = 0" and add "var_to_val[var] = val"
        # if the statement is of another form, track what variables show up on the rhs, and count how many times. if the var is in var_freq, update count
        
        var_freq = {}
        var_to_val = {}
        var_to_var = {}
        new_q = p.question
        for v in p.variables:
            if _is_numeric_constant(str(v.value)) or isinstance(v.value, Var):
                var_freq[str(v.name)] = 0
                var_to_val[str(v.name)] = v.value
                # if var to var situation (how do we check this), add var pairing to var_to_var
                if isinstance(v.value, Var):
                    var_to_var[v.name] = v.value
            else:
                for vf in var_freq.keys():
                    var_freq[vf] += str(v.value).count(vf)
        # loop through variable declarations in the problem
        # check if the statement is of form "var = num" and if var_freq[var] = 1" if yes, remove that statement from the problem
        # if the statement is of the form "var = something longer", find all vars in rhs, and see if any need to be replaced with their numerical value
        new_vars = []
        propagated_constants = set()
        for v in p.variables:
            if v.name in var_freq:
                if var_freq.get(str(v.name)) <= 1: # or if str(v.name) in var_to_var.keys()
                    propagated_constants.add(str(v.name))
            if isinstance(v.value, str): # only hit this case when running unit tests
                # String-replace all propagated constants and add to new_vars.
                new_value = v.value
                for pc in propagated_constants:
                    new_value = new_value.replace(pc, str(var_to_val[pc]))
                new_vars.append(VariableDeclaration(v.name, new_value))
            else:
                if v.name not in var_to_var and v.name not in propagated_constants:
                    new_vars.append(VariableDeclaration(v.name, replace_constants(v.value)))
        if str(p.question.question[0]) in var_to_var:
            x = replace_constants(p.question.question[0])
            new_q = QuestionDeclaration(p.question.name, [str(x)])
        new_p = Problem(new_vars, new_q, p.standard)
        return new_p


class NoTransform(ProblemTransformation):
    def __init__(self):
         self.name = "No Transform"

    def transform(self, p: Problem) -> Problem:
         return p


if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = ArgumentParser(description='Generate data based on grammar.')

    # Add the --generate flag to indicate generate mode
    parser.add_argument('--generate', action='store_true', help='Enable generate mode')
    
    # Add the --include_bad_problems flag to indicate that non-cycle-consistent problems shouldn't be removed
    parser.add_argument('--include_bad_problems', action='store_true', help='Includes problems that dont pass cycle consistency.')
    
    # Add the --use_claude flag to generate problems with claude instead of gpt
    parser.add_argument('--use_claude', action='store_true', help='Uses claude to generate problems.')

    # Add the --standard option followed by a standard ID
    parser.add_argument('--standard', type=str, help='Common Core Standard ID for generation')

    # Add the --n option followed by a positive integer
    parser.add_argument('--n', type=int, help='Number of problems to generate (positive integer)')

    # Add the --outputfile option followed by the output file path
    parser.add_argument('--outputfile', type=str, help='Output file path for the dataset')

    # Add the --fup option followed by number of follow up problems to generate
    parser.add_argument('--n_fup', type=int, help='Number of follow up problems to generate')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if generate mode is enabled
    if args.generate:
        # Check if the standard ID is provided
        if not args.standard:
            raise ValueError('Error: --standard is required in generate mode.')

        # Check if the number of items is provided and is a positive integer
        if args.n is not None:
            if args.n <= 0:
                raise ValueError('Error: --n must be a positive integer.')

        # Check if the dataset file path is provided
        if not args.outputfile:
            raise ValueError('Error: --outputfile is required in generate mode. Must be a json.')

        if args.n_fup is not None:
            if args.n_fup <= 0:
                raise ValueError('Error: --n must be a positive integer.')

        # If all checks pass, proceed with the rest of the code
        print('Generate mode is enabled.')
        print(f'Standard ID: {args.standard}')
        print(f'Number of items to generate: {args.n}')
        print(f'Dataset file path: {args.outputfile}')
        if args.n_fup is not None:
            print(f'Number of follow up problems to generate: {args.n_fup}')
    else:
        raise ValueError('Error: Please enable generate mode using --generate.')

    standards = load_standards()
    print("Standards implemented: ", standards.keys())

    """
    uncommenting the line below serves as a test to check if symbolic structures
    for all current standards can be properly generated without running into errors.
    """
    # generate_all_sym_structs(standards, args.n, args.n_fup)
    for standard in standards:
        data = []
        print(standards[standard].id)
        standard = standards[standard]
        print(len(standards))

        if standard.id not in ['K.CC.C.7', 'K.OA.A.4']:
            # standard = standards[args.standard]

            g_problems = generate_many(4*args.n, standard, args.n_fup)
            
            print("Standard", str(standard))
            for i, g_problem in enumerate(g_problems):
                p_seq = [ (0, g_problem.problem) ]
                if g_problem.followup_problems is not None:
                    p_seq.extend(g_problem.followup_problems)


                g_problem.theme = generate_theme(g_problem.problem)
                
                p_seq_data = []
                messages_to_nl = [
                    {"role": "system", "content": "You are an assistant that generates math problems based on a specified theme and symbolic structure. The math problem you generate must follow the operations provided in the symbolic structure. Do not simplify any of the expressions in the symbolic structure. For example, if it contains something like 'var x = 2 + 3', the problem you generate must contain both 2 and 3, and use addition in between them. Do not use the variable names from the symbolic structure in the problem."}
                ]
                messages_to_sym = [
                    {"role": "system", "content": """You are an assistant that generates symbolic structures from a given math problem. The symbolic structure must accurately reflect all steps in the math problem, and result in the correct final answer. Do not generate a theme or math concept in your symbolic structure. The only thing you return should be the symbolic structure. It is possible that the symbolic structure consists of only a question. Don't repeat variable names in the symbolic structure. All lines in the symbolic structure except the last one will use the 'var' keyword. The last line will use the 'question' keyword. Do not use any other keywords. If there are multiple answers (like in a system of equations), then the variables on the right hand side of the equal to symbol in the question should be a list where each variable is in quotes. If the problem is a followup problem to a problem above it, also incorporate the information from the original problem into the new symbolic structure.
                        If the problem does not have complete information, do not generate a symbolic structure."""}
                ]
                
                for x in range(int(len(standard.samples)/2 + 1)):
                    messages_to_nl.append({"role": "user", "content": standard.samples[x]})
                    messages_to_nl.append({"role": "assistant", "content": standard.samples[x + 1]})
                    messages_to_sym.append({"role": "user", "content": standard.samples[x + 1]})
                    messages_to_sym.append({"role": "assistant", "content": standard.samples[x]})

                # adding the "bad" problem to messages_to_sym
                messages_to_sym.append({"role": "user", "content": "In a forest, each tree has 16 birds. If these birds decide to evenly divide themselves among 8 trees, how many birds will be on each tree?"})
                messages_to_sym.append({"role": "assistant", "content": "This is a bad problem, as information about how many total trees are in the forest was not provided. Without that, it is impossible to create a symbolic structure to accurately reflect the problem."})
                
                for j, problem in p_seq:
                    input_problem = str(problem) + "\n" + g_problem.theme

                    print(f"Problem #{i+1}-{j+1}")
                    print(input_problem)

                    final_answer = calculate_final_answer(problem)
                    print("Final answer:", final_answer)
                    
                    #

                    if j == 0:
                        tag = "original problem"
                        messages_to_nl.append({"role": "user", "content": input_problem})
                        copy1_messages_to_nl = messages_to_nl
                        copy2_messages_to_nl = messages_to_nl
                    elif j == 1:
                        tag = "additional information follow up"
                        # in any generatedProblem, the first followup is an add_to_problem
                        copy1_messages_to_nl.append({"role": "assistant", "content": word_problem})
                        copy1_messages_to_nl.append({"role": "user", "content": "Consider the following structural change to the last problem:" + input_problem + "\n" + "Translate this into a follow-up question to the previous word problem, incorporating the last line in the new structure as an addition to the previous word problem." })
                        messages_to_nl = copy1_messages_to_nl
                    else:
                        tag = "modified information follow up"
                        # in any generatedProblem, the second followup is a modify_problem
                        copy2_messages_to_nl.append({"role": "assistant", "content": word_problem})
                        copy2_messages_to_nl.append({"role": "user", "content": "Consider the following structural change to the last problem:" + input_problem + "\n" + "Translate this into a follow-up question to the previous word problem. For example, if the previous problem reflected 5 + 2 by adding 5 oranges and 2 apples, and the new symbolic structure is 3 + 2, the follow-up question should be about considering what happens if originally, there were 3 oranges instead of 5." })
                        messages_to_nl = copy2_messages_to_nl
                    """
                    print("MESSAGES TO NL")
                    for message in messages_to_nl:
                        print(message)
                    """
                    if args.use_claude:
                        client = Anthropic()
                        message = client.messages.create(
                            model="claude-3-opus-20240229",
                            max_tokens=1000,
                            system=messages_to_nl[0]['content'],
                            messages=messages_to_nl[1:]
                        )
                        word_problem = message.content[0].text
                    else:
                        response = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages_to_nl
                        )
                        word_problem = response.choices[0].message.content
                    print(word_problem)
                    """
                    if j == 0:
                        messages_to_sym.append({"role": "user", "content": word_problem})
                        copy1_messages_to_sym = messages_to_sym
                        copy2_messages_to_sym = messages_to_sym
                    else if j == 1:
                        copy1_messages_to_sym.append(
                    else:
                    """
                    messages_to_sym.append({"role": "user", "content": word_problem})
                    """
                    print("MESSAGES TO SYM")
                    for message in messages_to_sym:
                        print(message)
                    """
                    # generating the symbolic structure based on the problem
                    if args.use_claude:
                        client = Anthropic()
                        message = client.messages.create(
                            model="claude-3-opus-20240229",
                            max_tokens=1000,
                            system=messages_to_sym[0]['content'],
                            messages=messages_to_sym[1:]
                        )
                        gpt_symbolic_structure = message.content[0].text
                    else:
                        response1 = client.chat.completions.create(
                            model="gpt-4",
                            messages=messages_to_sym
                        )
                        gpt_symbolic_structure = response1.choices[0].message.content
                    if j == 0:
                        messages_to_sym.append({"role": "assistant", "content": gpt_symbolic_structure})
                    print("new symbolic structure: \n", gpt_symbolic_structure)

                    gpt_symbolic_structure = gpt_symbolic_structure.rstrip('\n')
                    if j != 0:
                        # remove followup question from messages to sym
                        messages_to_sym = messages_to_sym[:-1]
                    # Check the symbolic structure
                    if standard.id not in ["4.NF.A.2", "K.CC.C.7", "4.OA.B.4"]:
                        if args.include_bad_problems:
                            p_seq_data.append({
                                "id": standard.id + "-" + str(i) + "-" + str(j),
                                "standard": standard.id,
                                "symbolic-struct": input_problem,
                                "statement": word_problem,
                                "new symbolic struct": gpt_symbolic_structure,
                                "cycle_consistent": check_sym_struct(gpt_symbolic_structure, final_answer, standard),
                                "answer": str(final_answer),
                                "bad_problem": None,
                                "tag": tag
                                # save even if new final answer doesnt match old, "cycle_consistent": t/f, add "bad_problem" field, set to null originally, manually edit
                                })
                        else:
                            if check_sym_struct(gpt_symbolic_structure, final_answer, standard):
                                p_seq_data.append({
                                    "id": standard.id + "-" + str(i) + "-" + str(j),
                                    "standard": standard.id,
                                    "symbolic-struct": input_problem,
                                    "statement": word_problem,
                                    "new symbolic struct": gpt_symbolic_structure,
                                    "answer": str(final_answer),
                                    "tag": tag
                                    })
                            else:
                                if j == 0:
                                    break
                    else:
                        if args.include_bad_problems:
                            p_seq_data.append({
                                "id": standard.id + "-" + str(i) + "-" + str(j),
                                "standard": standard.id,
                                "symbolic-struct": input_problem,
                                "statement": word_problem,
                                "new symbolic struct": gpt_symbolic_structure,
                                "cycle_consistent": True,
                                "answer": str(final_answer),
                                "bad_problem": None,
                                "tag": tag
                                # save even if new final answer doesnt match old, "cycle_consistent": t/f, add "bad_problem" field, set to null originally, manually edit
                                })
                        else:
                            p_seq_data.append({
                                "id": standard.id  + "-" + str(i) + "-" + str(j),
                                "standard": standard.id,
                                "symbolic-struct": input_problem,
                                "statement": word_problem,
                                "new symbolic struct": gpt_symbolic_structure,
                                "answer": str(final_answer),
                                "tag": tag
                            })
                if p_seq_data != []:
                    data.append(p_seq_data)
                if len(data) >= args.n:
                    break
            # json_file_path = args.outputfile
            json_file_path = f"{standard.id}.json"
            write_to_json(data, json_file_path)

    
