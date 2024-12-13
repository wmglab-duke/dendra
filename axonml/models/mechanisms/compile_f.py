import ast
import textwrap
import inspect

import ast

def transform_function(source: str) -> str:
    # Parse the source into an AST
    tree = ast.parse(source)
    
    # Find the function definition (assumes there is only one)
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    
    if func_def is None:
        raise ValueError("No function definition found in source.")

    func_name = func_def.name

    # Modify the function arguments to (self, v)
    func_def.args.args = [ast.arg(arg='self'), ast.arg(arg='v')]

    # First pass: Identify classification of variables.
    first_appearance = {}  # var_name: 'LHS' or 'RHS'
    def record_var_appearance(var_name: str, context: str):
        if var_name in ('self', 'v'):
            return
        if var_name not in first_appearance:
            first_appearance[var_name] = context

    # Visitor to record variable appearances
    class VarVisitor(ast.NodeVisitor):
        def __init__(self):
            super().__init__()
            self.in_lhs = False

        def visit_Assign(self, node):
            # Targets are LHS
            for target in node.targets:
                self.in_lhs = True
                self.visit(target)
                self.in_lhs = False
            # Value is RHS
            self.visit(node.value)
        
        def visit_AugAssign(self, node):
            # target is LHS
            self.in_lhs = True
            self.visit(node.target)
            self.in_lhs = False
            # value is RHS
            self.visit(node.value)

        def visit_Name(self, node):
            context = 'LHS' if self.in_lhs else 'RHS'
            record_var_appearance(node.id, context)

    var_visitor = VarVisitor()
    for stmt in func_def.body:
        var_visitor.visit(stmt)

    # Determine local and global sets
    local_vars = {var for var, ctx in first_appearance.items() if ctx == 'LHS'}
    global_vars = {var for var, ctx in first_appearance.items() if ctx == 'RHS'}

    def prepend_self_to_names(node):
        if isinstance(node, ast.Name):
            var_name = node.id
            # 'self' and 'v' remain as is.
            if var_name in ('self', 'v'):
                return node
            # If var is local, do not prefix
            if var_name in local_vars:
                return node
            # If var is global, prefix with self.
            if var_name in global_vars:
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=var_name, ctx=node.ctx),
                    node
                )
            # Otherwise, prefix with self
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=var_name, ctx=node.ctx),
                node
            )

        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, ast.AST):
                        new_list.append(prepend_self_to_names(item))
                    else:
                        new_list.append(item)
                setattr(node, field, new_list)
            elif isinstance(value, ast.AST):
                setattr(node, field, prepend_self_to_names(value))
        return node

    new_body = []
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign):
            stmt.value = prepend_self_to_names(stmt.value)
            new_body.append(stmt)

        elif isinstance(stmt, ast.Return):
            return_value = stmt.value
            if isinstance(return_value, ast.Name):
                return_target = return_value.id
            else:
                # Assign complex return to a temp variable first
                temp_name = "_return_temp"
                assign_node = ast.Assign(
                    targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                    value=prepend_self_to_names(return_value)
                )
                new_body.append(assign_node)
                return_target = temp_name
            
            # Assign return_target to self.<func_name>_
            new_body.append(
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                           attr=func_name + '_', ctx=ast.Store())],
                    value=ast.Name(id=return_target, ctx=ast.Load())
                )
            )

            # Return self.<func_name>_
            new_body.append(
                ast.Return(
                    value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                        attr=func_name + '_', ctx=ast.Load())
                )
            )
        else:
            new_body.append(prepend_self_to_names(stmt))

    func_def.body = new_body

    # Fix missing locations so that ast.unparse() doesn't throw errors
    ast.fix_missing_locations(tree)

    return ast.unparse(tree)


def convert_func(f):
    source = textwrap.dedent(inspect.getsource(f))
    return transform_function(source)