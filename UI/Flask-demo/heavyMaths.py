def listMultiply(a,b):
	a = list(a)
	b = int(b)
	print(a)
	print(b)
	newlist = []

	for element in a:
		newlist.append(int(element)*b)
	print(newlist)
	return newlist