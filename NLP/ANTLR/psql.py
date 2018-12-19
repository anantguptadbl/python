import pandas as pd
import numpy as np

import antlr4
from antlr4 import *
from psqlListener import psqlListener
from psqlLexer import psqlLexer
from psqlParser import psqlParser
import sys

class psqlPrintListener(psqlListener):
    def enterColNameList(self, ctx):
        columnNames=[]
        for curElement in ctx.children:
            if curElement.symbol.text != ",":
                columnNames.append(curElement.symbol.text)
        #print("ColumnName is %s" % ctx.NONSPACECONTINUOUS())
        print("The column names are {}".format(columnNames))
        
    def enterExpression(self, ctx):
        print(ctx)

inputStream=antlr4.InputStream('select col1,col2')
lexer = psqlLexer(inputStream)
stream = CommonTokenStream(lexer)
parser = psqlParser(stream)
tree=parser.expression()
listenerVal=psqlListener()
printer=psqlPrintListener()
walker=antlr4.ParseTreeWalker()
walker.walk(printer, tree)
