class MyTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        doc_string_found = 0
        for cur_element in node.body:
            #print("Line No and col Offset")
            #print(cur_element.lineno)
            #print(cur_element.col_offset)
            if isinstance(cur_element, ast.Expr) == True:
                if isinstance(cur_element.value, ast.Str) == True:
                    print(cur_element.value.s)
                    doc_string_found = 1
        # Now we will have to inject the doc string
        s = ast.Str("Injected a new docstring")
        new_expr = ast.Expr(value=s)
        if doc_string_found == 0:
            node.body.insert(0, new_expr)
        return node

expr = """
def add(arg1, arg2): 
    a=1
    # This is another simple comment
    return arg1 + arg2
"""
expr_ast = ast.parse(expr)
#print(ast.dump(expr_ast))
unmodified = ast.parse(expr)
exec(compile(unmodified, '<string>', 'exec'))
transformer = MyTransformer()
modified = transformer.visit(unmodified)
ast.fix_missing_locations(modified)
print("New String of the function")
print(astunparse.Unparser(modified, sys.stdout))
exec(compile(modified, '<string>', 'exec'))
print(add.__doc__)
