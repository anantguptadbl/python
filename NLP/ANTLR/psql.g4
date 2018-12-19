grammar psql;
expression : SELECT (WS)* colNameList;
SELECT : 'select' | 'SELECT' | 'Select';
FROM : 'FROM' | 'from' | 'From';
colNameList : (NONSPACECONTINUOUS (WS)* COMMA)* (WS)* NONSPACECONTINUOUS;
NONSPACECONTINUOUS : ('a'..'z'|'A'..'Z'|'0'..'9')+;
WS : ' ' -> skip;
COMMA : ',';