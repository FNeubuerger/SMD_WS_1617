x = 0
t = 0

for i in range(23):
	t = 2**(-1-i)
#	print(str(i))
	if (t + x) <= 2/3:
		print(str(t+x))
		x = t + x
#		print(1)
#	else:
#		print(0)
