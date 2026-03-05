from src.patchsim.core import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



casesfile = "IRDD_allka.csv"
casesframe = pd.read_csv(casesfile)
districts = casesframe['District'].unique()
age_stratification = [i*5 for i in range(80//5)]
print(age_stratification)
agegroups_min = np.array((0, 15, 65 ))

print(casesframe.head(500))
print(casesframe[casesframe['District']=='Bagalakote'])
print(f'Number of distrcits: {len(districts)}')
