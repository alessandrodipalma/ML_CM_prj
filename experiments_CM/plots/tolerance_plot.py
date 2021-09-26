from matplotlib import pyplot as plt
d = [2.86E-11, 2.84E-11, 4.78E-11, 1.68E-09, 1.94E-07, 2.81E-06, 5.50E-05, 1.05E-03]
f = [1.04E-07, 1.90E-06, 3.80E-06, 2.12E-05, 1.62E-04, 1.22E-03, 1.49E-02, 1.66E-01]
x = [3.52E-08, 9.03E-08, 2.75E-06, 7.26E-06, 2.01E-04, 2.37E-03, 2.34E-02, 3.15E-01]

y = [1.00E-08, 1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 1.00E-02, 1.00E-01]

plt.plot(y, d, label="gradient norm")

plt.plot(y, f, label="relative gap")
plt.plot(y, x, label="relative error")
plt.loglog()
plt.legend()
plt.xlabel("tolerance")
plt.ylabel("real gap")
plt.show()