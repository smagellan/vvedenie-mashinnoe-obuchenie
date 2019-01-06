#! /usr/bin/python3
import pandas as pd;


def guessFirstName(name):
    if '(' in name:
        pIndex1    = name.find('(');
        spaceIndex = name.find(' ', pIndex1);
        pIndex2    = name.find(')', pIndex1);
        endIndex   = spaceIndex if spaceIndex > 0 else pIndex2;
        return name[pIndex1 + 1:endIndex].strip();
    if '.' in name:
        dIndex     =  name.find('.');
        spaceIndex =  name.find(' ', dIndex + 2);
        return name[dIndex + 2:spaceIndex].strip();
    return name;

data = pd.read_csv('titanic.csv', index_col='PassengerId');


hm = dict();
for nm in data[data['Sex'] == 'female']['Name']:
    firstName = guessFirstName(nm);
    if firstName in hm:
        hm[firstName] = hm[firstName] + 1;
    else:
        hm[firstName] = 1;

for nm in hm:
    cnt = hm[nm];
    if cnt > 5:
        print("{}:{}".format(nm, cnt));
