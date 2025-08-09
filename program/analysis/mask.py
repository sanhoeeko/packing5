import re

import numpy as np

from . import utils as ut
from .voronoi import Voronoi


class Mask:
    """
    Mask class that supports logical operations (AND, OR, NOT)
    and can be called with (abg, xyt) to produce a boolean mask.
    """

    def __init__(self, mask_func):
        """
        mask_func: callable that takes (abg, xyt) and returns a boolean mask array
        """
        self.mask_func = mask_func

    def __call__(self, abg, xyt):
        return self.mask_func(abg, xyt)

    def __and__(self, other):
        def new_func(abg, xyt):
            return self(abg, xyt) & other(abg, xyt)

        return Mask(new_func)

    def __or__(self, other):
        def new_func(abg, xyt):
            return self(abg, xyt) | other(abg, xyt)

        return Mask(new_func)

    def __invert__(self):
        def new_func(abg, xyt):
            return ~self(abg, xyt)

        return Mask(new_func)


def parse_mask_expr(expr: str) -> Mask:
    """
    解析逻辑表达式字符串并返回对应的 Mask 对象
    支持运算符: ~ (非), | (或), & (与), 以及括号分组
    标识符映射到 MaskPrefab 中的静态方法

    示例:
        parse_mask_expr("~(a|b)&c")
        等价于 (~(a | b)) & c
    """
    # 定义运算符优先级
    PRECEDENCE = {'~': 3, '&': 2, '|': 1}

    # 标记化：将表达式拆分为令牌 (标识符、运算符、括号)
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|~|&|\||\(|\)', expr)

    # 处理错误：空表达式
    if not tokens:
        raise ValueError(f"Empty mask expression: {expr}")

    # 创建两个栈：运算符栈和操作数栈
    op_stack = []
    mask_stack = []

    # 辅助函数：应用运算符
    def apply_operator(op):
        if op == '~':
            if not mask_stack:
                raise ValueError(f"Missing operand for operator '{op}' in: {expr}")
            operand = mask_stack.pop()
            mask_stack.append(~operand)
        else:  # 二元运算符: & 或 |
            if len(mask_stack) < 2:
                raise ValueError(f"Missing operands for operator '{op}' in: {expr}")
            right = mask_stack.pop()
            left = mask_stack.pop()
            if op == '&':
                mask_stack.append(left & right)
            elif op == '|':
                mask_stack.append(left | right)

    # 遍历所有令牌
    for token in tokens:
        if token in MaskPrefab.__dict__:  # 标识符 (如 "boundary")
            # 获取对应的 Mask 对象
            mask_func = getattr(MaskPrefab, token)
            if callable(mask_func):
                mask_stack.append(mask_func())
            else:
                raise ValueError(f"'{token}' is not a valid mask function in MaskPrefab")

        elif token == '(':
            op_stack.append(token)

        elif token == ')':
            # 处理括号内的所有运算符
            while op_stack and op_stack[-1] != '(':
                apply_operator(op_stack.pop())
            if not op_stack or op_stack[-1] != '(':
                raise ValueError(f"Mismatched parentheses in: {expr}")
            op_stack.pop()  # 弹出 '('

        elif token in PRECEDENCE:  # 运算符: ~, &, |
            # 处理更高或相同优先级的运算符
            while (op_stack and op_stack[-1] != '(' and
                   PRECEDENCE.get(op_stack[-1], 0) >= PRECEDENCE[token]):
                apply_operator(op_stack.pop())
            op_stack.append(token)

    # 处理剩余的运算符
    while op_stack:
        op = op_stack.pop()
        if op == '(':
            raise ValueError(f"Mismatched parentheses in: {expr}")
        apply_operator(op)

    # 结果应在 mask_stack 中
    if len(mask_stack) != 1:
        raise ValueError(f"Invalid expression structure: {expr}")

    return mask_stack[0]


class MaskPrefab:
    @staticmethod
    def body():
        def mask_func(abg, xyt):
            A, B, gamma = abg
            xyt_c = ut.CArray(xyt, dtype=np.float32)
            voro = Voronoi(gamma, A, B, xyt_c.data).delaunay()
            return ~voro.dist_hull(xyt_c).astype(bool)

        return Mask(mask_func)

    @staticmethod
    def internal():
        """Create mask for internal particles"""

        def mask_func(abg, xyt):
            N = xyt.shape[0]
            A, B, gamma = abg
            phi = ut.phi(N, gamma, A, B)
            return ut.InternalMask2(phi, A, B, xyt)

        return Mask(mask_func)

    @staticmethod
    def boundary():
        return ~MaskPrefab.internal()

    @staticmethod
    def y_rank():
        """Create mask based on y rank"""

        def mask_func(abg, xyt):
            N = xyt.shape[0]
            A, B, gamma = abg
            phi = ut.phi(N, gamma, A, B)
            return ut.y_rank(N, phi, xyt, B)

        return Mask(mask_func)

    @staticmethod
    def _general_dense_mask(flag):
        """Create mask based on density flags"""

        def mask_func(abg, xyt):
            A, B, gamma = abg
            xyt_c = ut.CArray(xyt, dtype=np.float32)
            voro = Voronoi(gamma, A, B, xyt_c.data).delaunay()
            dense = voro.dense(xyt_c)
            return dense == flag

        return Mask(mask_func)

    @staticmethod
    def dense0():
        return MaskPrefab._general_dense_mask(0)

    @staticmethod
    def dense1():
        return MaskPrefab._general_dense_mask(0)

    @staticmethod
    def dense2():
        return MaskPrefab._general_dense_mask(0)
