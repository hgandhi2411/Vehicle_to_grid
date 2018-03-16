import numpy as np
import datetime
import pandas as pd


nyc_data = np.genfromtxt('LBMP/nyc.csv', delimiter=',', skip_header=1, dtype=np.object, skip_footer=1)
last = None
to_insert = []
to_remove = []
for i, date in enumerate(nyc_data[:,0].astype(str)):
    dt = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M')
    if last is not None:
        if (dt - last).seconds // 3600 == 0:
            to_remove.append(i)
        else:
            for j in range((dt - last).seconds // 3600 - 1):            
                #add for as many missing hours
                to_insert.append(i)
    last = dt
print('Off by {}'.format(nyc_data.shape[0] % 24))
print(f'Found missing data at {to_insert}')
print(f'Found duplicate data at {to_remove}')
print('Will be off by {}'.format((nyc_data.shape[0] - len(to_remove) + len(to_insert)) % 24))
for i,j in enumerate(to_insert):
    for k in to_remove:
        if j > k:
            to_insert[i] += 1
print(f'Removing {to_remove}')
to_remove.sort()
for i in to_remove[::-1]:
    nyc_data = np.delete(nyc_data, i, axis=0)
print(f'Inserting {to_insert}')
to_insert.sort()
for i in to_insert:
    nyc_data = np.insert(nyc_data, i, nyc_data[i], axis=0)

nyc_data = pd.DataFrame(nyc_data)
nyc_data.to_csv('LBMP/nyc.csv')