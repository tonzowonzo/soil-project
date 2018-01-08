# Trying to classify soil based on world soil data
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load in csv dataset
df = pd.read_excel('HWSD_DATA.xlsx')
df = df[df['ISSOIL'].isin([1])] # Drop all the non-soils
df = df[df.SU_CODE90.notnull()] # Drop all non-classified soils.
categorical_vars = ['SU_CODE74', 'SU_CODE85', 'SU_CODE90', 'T_TEXTURE', 'DRAINAGE', 'AWC_CLASS',
                    'ROOTS', 'IL', 'SWR', 'ADD_PROP']
# Get rid of all of the useless columns
df = df.iloc[:, 7:]

# Fill NaN values for amount of silt, sand, gravel and clay.
df['T_CLAY'].fillna(df['T_CLAY'].median(), inplace=True)
df['T_SILT'].fillna(df['T_SILT'].median(), inplace=True)
df['T_SAND'].fillna(df['T_SAND'].median(), inplace=True)
df['T_GRAVEL'].fillna(df['T_GRAVEL'].median(), inplace=True)

# Add new feature of absolute soil class
df['SOIL_CLASS'] = [x[:2] for x in df['SU_SYM90']]
# Amount of each type of soil
'''
Below shows that most our soil types have sample sizes of <20 and are largely spread from
0 to 200, there is one soil type I which almost contains 1000 samples however.
'''
df['SU_SYM90'].unique()
x = df['SU_SYM90'].value_counts()
plt.hist(x, bins=20)

'''
Soil class value counts, similar to above but for absolute classes instead of minimum.
'''
df['SOIL_CLASS'].unique()
x = df['SOIL_CLASS'].value_counts()
plt.hist(x)
'''
Below shows the distribution of drainage types with 1 being very poor drainage and
6 being somewhat excessive. The most common class is 4 which is a moderately-well drained
soil.
'''
plt.hist(df['DRAINAGE'], bins=6)

'''
Below shows the distribution of the texture types. 2 is the most common texture by
a significant margin, 2 is a medium texture.
'''
plt.hist(df['T_TEXTURE'], bins=4)

'''
Below shows the distributions of each type of soil particulate matter. Gravel is rarely
over 10% in soils while sand commonly makes up over 30% of a soil.
'''
sns.pairplot(data=df, vars=['T_CLAY', 'T_SILT', 'T_SAND', 'T_GRAVEL'])
plt.hist(df['T_CLAY'].dropna(), bins=20)
plt.hist(df['T_SILT'].dropna(), bins=20)
plt.hist(df['T_SAND'].dropna(), bins=20)
plt.hist(df['T_GRAVEL'].dropna(), bins=20)



