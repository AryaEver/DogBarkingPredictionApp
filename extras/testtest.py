import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.asarray([1,2,3,4,5])
b = np.asarray([[5],[6],[7],[4],[9]])

c = a*b

#plt.imshow(c)


datos = pd.read_csv('./mescalina2015_500LLDs.csv').as_matrix()
datos = datos.astype(str)
