# Some Examples
#https://github.com/AlanHohn/antlr4-python

import pandas as pd
import numpy as np

from antlr4 import *
from psqlListener import psqlListener
from psqlLexer import psqlLexer
from psqlParser import psqlParser
import sys

lexer = psqlLexer("select col1 from table1")
stream = CommonTokenStream(lexer)
parser = psqlParser(stream)
tree=parser.operation()
listenerVal=psqlListener()

class psqlPrintListener(psqlListener):
    def enterColumnNames(self,ctx):
        columnNameContext=ctx.columnName()
        
        
    def enterColumnName(self, ctx):
        print("ColumnName is %s" % ctx.NONSPACECONTINUOUS())
        
    def enterOperation(self, ctx):
        print(ctx)
        
printer=psqlPrintListener()
walker=antlr4.ParseTreeWalker()
walker.walk(printer, tree)
