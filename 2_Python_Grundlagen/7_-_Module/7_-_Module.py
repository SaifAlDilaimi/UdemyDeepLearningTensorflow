#### Module in Python ####

# from DATEI import FUNKTION
#from students import bester_schüler
from students import *

# import DATEI
#import students as st

noten_klasse_10a = {"armin": 78, "ben": 89, "jan": 84, "peter": 99}

name, note = bester_schüler(noten_klasse_10a, konsolenausgabe=True)

import random

zahl = random.randint(1, 10)
print(zahl)