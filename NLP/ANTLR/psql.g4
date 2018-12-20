grammar psql;
expression : SELECT (WS)* colNameList (WS)* FROM (WS)* tableName (WS)*;
SELECT : 'select' | 'SELECT' | 'Select';
FROM : 'FROM' | 'from' | 'From';
colNameList : (NONSPACECONTINUOUS (WS)* COMMA)* (WS)* NONSPACECONTINUOUS;
tableName : NONSPACECONTINUOUS;
NONSPACECONTINUOUS : ('a'..'z'|'A'..'Z'|'0'..'9')+;
WS : ' ' -> skip;
COMMA : ',';