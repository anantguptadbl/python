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
