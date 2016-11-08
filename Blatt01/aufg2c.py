import numpy as np
import ROOT
#import math dann braucht man das auch nciht

Zufallszahlen = np.random.randint(1, 100, 100000)
#LogZahl = np.zeros(100000) Python is dynamisch, das heißt wir müssen keine leeren arrays initialisieren. Laufzeittechnisch ist es evtl sinnvoll dass gewusst wird wie viel speicher man braucht aber das is bei python sowieso egal.

#i = 0
#while i < 100000:
#	LogZahl[i] = math.log(Zufallszahlen[i]) #while Schleifen in python sind nicht nötig. numpy kann arrays elementweise zu neuen arrays machen und die rechenoperation anwenden. Sehr praktisch
#	i += 1

LogZahlen = np.log(Zufallszahlen)


Blatt_Zufallszahl = ROOT.TCanvas("Zufallszahlenhistogramme", "Zufallszahlenverteilung", 800, 1200)

Zufallszahl_H1 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 5)", 5, 0 , np.log(100)+1) #die letzetn argumente in der FKT sind die x grenzen, wie ich das aus der root doku finden konnte
Zufallszahl_H2 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 10)", 10, 0, np.log(100)+1)
Zufallszahl_H3 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 15)", 15, 0, np.log(100)+1)
Zufallszahl_H4 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 20)", 20, 0, np.log(100)+1)
Zufallszahl_H5 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 30)", 30, 0, np.log(100)+1)
Zufallszahl_H6 = ROOT.TH1F("Zufallszahl", "Zufallszahlenvert. (Bin 50)", 50, 0, np.log(100)+1)
for i in LogZahlen: #Zufallszahlen
	Zufallszahl_H1.Fill(i)
	Zufallszahl_H2.Fill(i)
	Zufallszahl_H3.Fill(i)
	Zufallszahl_H4.Fill(i)
	Zufallszahl_H5.Fill(i)
	Zufallszahl_H6.Fill(i)


Blatt_Zufallszahl.Divide(2,3)

Blatt_Zufallszahl.cd(1)
Zufallszahl_H1.Draw()

Blatt_Zufallszahl.cd(2)
Zufallszahl_H2.Draw()
Blatt_Zufallszahl.cd(3)
Zufallszahl_H3.Draw()
Blatt_Zufallszahl.cd(4)
Zufallszahl_H4.Draw()
Blatt_Zufallszahl.cd(5)
Zufallszahl_H5.Draw()
Blatt_Zufallszahl.cd(6)
Zufallszahl_H6.Draw()


Blatt_Zufallszahl.Update()

Blatt_Zufallszahl.SetLogy()
Blatt_Zufallszahl.SaveAs("Log_ZufallszahlenverteilungBinnings2x3.png")