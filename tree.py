#!/usr/bin/env python3


from dataclasses import dataclass


class Node:
    '''Base class for all nodes in the tree.'''
    def children(self) -> list['Node']:
        '''Returns a list of children of this node.'''
        raise NotImplementedError
    
    def set_child(self, index: int, node: 'Node'):
        '''Sets the i-th child of this node.'''
        raise NotImplementedError



class Expr(Node):
    '''Base class for expression nodes.'''
    def __eq__(self, e2):
        return str(self) == str(e2)
    
    def subtree_nodes(self):
        raise NotImplementedError


@dataclass
class VariableDeclaration(Node):
    'A declaration of a variable with a given expression as its value.'
    name: str
    value: 'Expr'

    def __str__(self) -> str:
        return f'[[var {self.name} = {self.value}]]'

    def noBracStr(self) -> str:
        return f'{self.name} = {self.value}'

    def children(self) -> list[Node]:
        return [self.value] + self.value.children()

    def set_child(self, index: int, node: Node):
        assert index == 0, 'VariableDeclaration only has 1 child.'
        self.value = node

    def subtree_nodes(self):
        return [self.value] + self.value.subtree_nodes()
        
    def replace_node(self, index, newNode):
        # print("self value", self.value, "index", index)
        self.value.replace_node(index, newNode)
        # print("new var dec value", self.value)
            
        


@dataclass
class QuestionDeclaration(Node):
    'A declaration of the expression that is the answer to the question.'
    name: str
    question: 'Expr'

    def __str__(self) -> str:
        return f'[[question {self.name} = {[str(q) for q in self.question]}]]'

    def children(self) -> list[Node]:
        return [self.question] + self.question.children()

    def set_child(self, index: int, node: Node):
        assert index == 0, 'QuestionDeclaration only has 1 child.'
        self.question = node
    
    def subtree_nodes(self):
        to_ret = []
        for q in self.question:
            to_ret.append(q)
            to_ret.extend(q.subtree_nodes())
        return to_ret
        
    def replace_node(self, index, newNode):
        """
        lhs_size = len(self.lhs.subtree_nodes()) + 1
        print("index", index)
        print("binop lhs subtree nodes", self.lhs.subtree_nodes())
        if index <= lhs_size:
            # print("in lhs", self.lhs)
            self.lhs.replace_node(index - 1, newNode)
            # print("binop after lhs val change", self.lhs)
        else:
            # print("in rhs", self.rhs)
            self.rhs.replace_node(index - lhs_size - 1, newNode)
            # print("binop after rhs val change", self.rhs)
        """
        # print("index", index)
        count = 0
        var_to_change = self.question[0]
        for q in self.question:
            # print("question", q)
            curr_subtree_nodes = q.subtree_nodes()
            # print("curr_subtree_nodes", curr_subtree_nodes, len(curr_subtree_nodes))
            count += len(curr_subtree_nodes)
            if index <= count:
                # print("count", count)
                var_to_change = q
                count -= len(curr_subtree_nodes)
                # print("final count", count)
                break
        # print("var to change", var_to_change)
        var_to_change.replace_node(index - count, newNode)



@dataclass
class Num(Expr):
    'A numeric constant.'
    value: int

    def __str__(self) -> str:
        return f'{self.value}'

    def children(self) -> list[Node]:
        return []

    def set_child(self, _index, _node):
        raise ValueError('Num has no children')
    
    def subtree_nodes(self):
        return []
    
    def __eq__(self, n2):
        return isinstance(n2, Num) and self.value == n2.value
        
    def replace_node(self, index, newNode):
        # print("self value before num change", self.value)
        # assert index == 0, f"index must be 0, but is currently {index}"
        self.value = newNode.value
        # print("self value after num change", self.value)
        
    def replace_to_var(self, index, newNode):
        print("num before", self)
        self = None
        self = newNode
        print("num after", self)
    
    def __hash__(self):
        return hash(self.value)

@dataclass
class Const(Expr):
    'A numeric constant.'
    value: int

    def __str__(self) -> str:
        return f'{self.value}'

    def children(self) -> list[Node]:
        return []

    def set_child(self, _index, _node):
        raise ValueError('Const has no children')
        
    def subtree_nodes(self):
        return []
    
    def replace_node(self, index, newNode):
        assert("Should not be replacing Const node")


@dataclass
class Var(Expr):
    'A reference to a previously declared variable.'
    name: str

    def __str__(self) -> str:
        return f'{self.name}'

    def children(self) -> list[Node]:
        return []

    def set_child(self, _index, _node):
        raise ValueError('Var has no children')
    
    def subtree_nodes(self):
        return []
        
    def replace_node(self, index, newNode):
        assert("Should not be replacing Var node")


@dataclass
class BinOp(Expr):
    'Abstract class for binary operations.'
    lhs: 'Expr'
    rhs: 'Expr'

    def __str__(self) -> str:
        raise NotImplementedError

    def children(self, node: Node = None) -> list[Node]:
        """Return children in a flattened, depth-first ordered array of self."""
        if node is None:
            node = self
        
        if isinstance(node, Var) or isinstance(node, Num) or isinstance(node, Const):
            return [node]
        elif isinstance(node, BinOp):
            return self.children(node.lhs) + self.children(node.rhs)
        else:
            assert False, "Invalid expression detected in children"

    def set_child(self, index: int, new_value, node: Node = None) -> (Node, int):
        """Set the i-th child of the flattened array of self. Note that child must be a Num or Var."""
        if node is None:
            node = self

        if index == 0:
            node = new_value
            return (node, index)
        elif isinstance(node, Num) or isinstance(node, Var) or isinstance(node, Const):
            return (node, index)
        elif isinstance(node, BinOp):
            if index - 1 >= 0:
                node.lhs, new_index_0 = self.set_child(index - 1, new_value, node.lhs)
            else:
                new_index_0 = index
            if new_index_0 - 1 >= 0:
                node.rhs, new_index_1 = self.set_child(new_index_0 - 1, new_value, node.rhs)
            else:
                new_index_1 = new_index_0
            return (node, new_index_1)
        else:
            assert False, "Invalid expression detected in set_child"
    
    def subtree_nodes(self):
        return [self.lhs] + self.lhs.subtree_nodes() + [self.rhs] + self.rhs.subtree_nodes()
    
    def replace_node(self, index, newNode):
        lhs_size = len(self.lhs.subtree_nodes()) + 1
        if index <= lhs_size:
            # print("in lhs", self.lhs)
            self.lhs.replace_node(index - 1, newNode)
            # print("binop after lhs val change", self.lhs)
        else:
            # print("in rhs", self.rhs)
            self.rhs.replace_node(index - lhs_size - 1, newNode)
            # print("binop after rhs val change", self.rhs)
    
    def replace_to_var(self, index, newNode):
        # print(self.lhs.subtree_nodes())
        # print("lhs subtree nodes", self.lhs.subtree_nodes())
        lhs_size = len(self.lhs.subtree_nodes())
        if index < lhs_size + 1:
            # [(3 + 5) + 2, (3 + 5), 3, 5, 2]
            # index = 2
            # lhs case -> self.lhs.r(1, newNode) -> [3, 5] -> replace 3
            # index = 3
            # lhs case -> self.lhs.r(2, newNode) -> [3, 5] ->
            # index = 4
            if index == 0:
                # print("lhs before", self.lhs)
                self.lhs = newNode
                # print("lhs after", self.lhs)
            else:
                if isinstance(self.lhs, BinOp):
                    self.lhs.replace_to_var(index - 1, newNode)
                    # print(type(self.lhs))
        else:
            if isinstance(self.rhs, BinOp):
                self.rhs.replace_to_var(index - lhs_size, newNode)
            else:
                self.rhs = newNode
            """
            if index == 0:
                print("rhs before", self.rhs)
                self.rhs = newNode
                print("rhs after", self.rhs)
            else:
                print("recursive rhs call with new index", index - lhs_size)
                self.rhs.replace_to_var(index - lhs_size, newNode)
            """

        

class Add(BinOp):
    def __str__(self) -> str:
        return f'({self.lhs} + {self.rhs})'


class Sub(BinOp):
    def __str__(self) -> str:
        return f'({self.lhs} - {self.rhs})'


class Mult(BinOp):
    def __str__(self) -> str:
        return f'({self.lhs} * {self.rhs})'


class Div(BinOp):
    def __str__(self) -> str:
        return f'({self.lhs} / {self.rhs})'


class Remainder(BinOp):
    def __str__(self) -> str:
        return f'({self.lhs} % {self.rhs})'

@dataclass
class Frac(BinOp):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.value = self.lhs.value/self.rhs.value
    def __str__(self) -> str:
        return f'({self.lhs} / {self.rhs})'

class Compare(BinOp):
    def __str__(self) -> str:
        return f'{self.lhs} _ {self.rhs}'

class Square(BinOp):

    def __init__(self, lhs):
        self.lhs = lhs
        self.rhs = Const(2)
    def __str__(self) -> str:
        return f'({self.lhs} ** {self.rhs})'


class Cube(BinOp):
    def __init__(self, lhs):
        self.lhs = lhs
        self.rhs = Const(3)
    def __str__(self) -> str:
        return f'({self.lhs} ** {self.rhs})'

class Exponent(BinOp):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def __str__(self) -> str:
        return f'({self.lhs} ** {self.rhs})'
