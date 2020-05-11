import numpy as np

mean_McDnCnn = 0.0195331603102386
std_McDnCnn = 0.003922510588794393
mean_CAE=  0.022717652171850206
std_CAE=  0.0038151210266976015

top = np.square((np.square(std_McDnCnn)/100) + (np.square(std_CAE)/100))

bot = 1/(99)*np.square(np.square(std_McDnCnn)/100)+1/(99)*np.square(np.square(std_CAE)/100)

print(top/bot)
