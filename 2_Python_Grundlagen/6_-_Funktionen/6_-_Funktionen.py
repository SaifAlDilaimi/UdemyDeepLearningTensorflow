#### Funktionen in Python ####

def bester_schüler(noten_klasse, konsolenausgabe=False):
    bis_jetzt_bester_schüler = ""
    bis_jetzt_beste_note = 0
    for name, note in noten_klasse.items():
        if note > bis_jetzt_beste_note:
            bis_jetzt_beste_note = note
            bis_jetzt_bester_schüler = name
    if konsolenausgabe == True:
        print("Ausgabe: ", bis_jetzt_bester_schüler, bis_jetzt_beste_note)
    return bis_jetzt_bester_schüler, bis_jetzt_beste_note




noten_klasse_10a = {"armin": 78, "ben": 89, "jan": 84, "peter": 99}

name, note = bester_schüler(noten_klasse_10a, konsolenausgabe=True)