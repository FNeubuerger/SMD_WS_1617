import numpy as np
import ROOT

Menge = np.random.random(20000000)
x0 = 1/2
count = 0

Menge = Menge.astype(np.float32)

for i in Menge:
	if i == x0:
		count = count + 1

print("Fuer Numpy Random:")
print("Der exakte Wert 1/2 ist genau " + str(count) + " Mal in 20.000.000 Zufallszahlen im Intervall [0,1]")


Menge2 = np.zeros(20000000)
count2 = 0

rdm = ROOT.TRandom()
rdm.RndmArray(20000000, Menge2)
Menge2 = Menge2.astype(np.float32)

for i in Menge2:
	if i == x0:
		count2 = count2 + 1


print("Fuer Root TRandom:")
print("Der exakte Wert 1/2 ist genau " + str(count2) + " Mal in 20.000.000 Zufallszahlen im Intervall [0,1]")

