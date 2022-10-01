import matplotlib.pyplot as plt

f = open("output.txt","r")
time = []
error = []
for x in f:
    xx = x.split(" ")
    time.append(int(xx[0]))
    error.append(float(xx[1]))

plt.plot(time,error)
plt.xlabel('epoch')
plt.ylabel('error')

plt.show()