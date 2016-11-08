import numpy as np
import ROOT


Gewicht, Groesse = np.genfromtxt('Groesse_Gewicht.txt', unpack=True)


#
# 6 Histogramme mit unterschiedlicher Anzahl von Binnings
#


# Datensaetze zur Groesse



Blatt_Groesse = ROOT.TCanvas("Groessenhistogramme", "Groessenverteilung", 800, 1200)

# linker und rechter Rand eventuell durch np.min(Groesse) und 
# np.max(Groesse) bestimmen.

Groesse_H1 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 5)", 5, 1.5, 2)
Groesse_H2 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 10)", 10, 1.5, 2)
Groesse_H3 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 15)", 15, 1.5, 2)
Groesse_H4 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 20)", 20, 1.5, 2)
Groesse_H5 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 30)", 30, 1.5, 2)
Groesse_H6 = ROOT.TH1F("Groesse", "Groessenvert. (Bin 50)", 50, 1.5, 2)


for i in Groesse:
	Groesse_H1.Fill(i)
	Groesse_H2.Fill(i)
	Groesse_H3.Fill(i)
	Groesse_H4.Fill(i)
	Groesse_H5.Fill(i)
	Groesse_H6.Fill(i)

Blatt_Groesse.Divide(2,3)

Blatt_Groesse.cd(1)
Groesse_H1.Draw()
Blatt_Groesse.cd(2)
Groesse_H2.Draw()
Blatt_Groesse.cd(3)
Groesse_H3.Draw()
Blatt_Groesse.cd(4)
Groesse_H4.Draw()
Blatt_Groesse.cd(5)
Groesse_H5.Draw()
Blatt_Groesse.cd(6)
Groesse_H6.Draw()



Blatt_Groesse.Update()
Blatt_Groesse.SaveAs("./GroessenverteilungBinnings2x3.png")




Blatt_Gewicht = ROOT.TCanvas("Gewichtshistogramme", "Gewichtsverteilung", 800, 1200)

# linker und rechter Rand eventuell durch np.min(Gewicht) und 
# np.max(Gewicht) bestimmen.

Gewicht_H1 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 05)", 5, 50, 200)
Gewicht_H2 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 10)", 10, 50, 200)
Gewicht_H3 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 15)", 15, 50, 200)
Gewicht_H4 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 20)", 20, 50, 200)
Gewicht_H5 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 30)", 30, 50, 200)
Gewicht_H6 = ROOT.TH1F("Gewicht", "Gewichtsvert. (Bin 50)", 50, 50, 200)


for i in Gewicht:
	Gewicht_H1.Fill(i)
	Gewicht_H2.Fill(i)
	Gewicht_H3.Fill(i)
	Gewicht_H4.Fill(i)
	Gewicht_H5.Fill(i)
	Gewicht_H6.Fill(i)

Blatt_Gewicht.Divide(2,3)

Blatt_Gewicht.cd(1)
Gewicht_H1.Draw()
Blatt_Gewicht.cd(2)
Gewicht_H2.Draw()
Blatt_Gewicht.cd(3)
Gewicht_H3.Draw()
Blatt_Gewicht.cd(4)
Gewicht_H4.Draw()
Blatt_Gewicht.cd(5)
Gewicht_H5.Draw()
Blatt_Gewicht.cd(6)
Gewicht_H6.Draw()



Blatt_Gewicht.Update()
Blatt_Gewicht.SaveAs("./GewichtsverteilungBinnings2x3.png")