grammar psql;
expression : SELECT colNameList;
SELECT : 'select' | 'SELECT' | 'Select';
FROM : 'FROM' | 'from' | 'From';
colNameList : STRINGVAL;
tableName : STRINGVAL;
STRINGVAL : ('a'..'z'|'A'..'Z'|'0'..'9')+;
