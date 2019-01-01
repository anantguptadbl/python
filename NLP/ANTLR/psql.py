# Running the command for generating the lexer and parser
# antlr4 -Dlanguage=Python2 psql.g4

import pandas as pd
import numpy as np

import antlr4
from antlr4 import *
from psqlListener import psqlListener
from psqlLexer import psqlLexer
from psqlParser import psqlParser
import sys
a=1
class psqlPrintListener(psqlListener):
    def enterColNameList(self, ctx):
        columnNames=[]
        for curElement in ctx.children:
            if curElement.symbol.text != ",":
                columnNames.append(curElement.symbol.text)
        #print("ColumnName is %s" % ctx.NONSPACECONTINUOUS())
        print("The column names are {}".format(columnNames))

    def enterTableName(self, ctx):
        print("The table name is {}".format(ctx.children[0].symbol.text))

    def enterExpression(self, ctx):
        global a
        a=ctx
        print(ctx)

inputStream=antlr4.InputStream('select col1,col2 from table1')
lexer = psqlLexer(inputStream)
stream = CommonTokenStream(lexer)
parser = psqlParser(stream)
tree=parser.expression()
listenerVal=psqlListener()
printer=psqlPrintListener()
walker=antlr4.ParseTreeWalker()
walker.walk(printer, tree)

# Some actual data
data=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['col1','col2','col3'])

# Parsing stuff
b=a.children[0]
#b.getText() # This will give select
#c=b.symbol
#d=c.getTokenSource()
#lexer.getVocabulary.getSymbolicName(b.getSymbol.getType)
c=b.getSymbol()
rules=lexer.ruleNames
tokenIndex=c.tokenIndex

