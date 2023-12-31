#side quest - seeing the top major and minor songs
print('all bottom 20')
allTop20 = data[['popularity', 'track_name', 'artists']]
allTop20 = allTop20.sort_values(by=['popularity'], ascending=True)
for i in range(20):
    print(allTop20.iloc[i]['track_name'], ' by ', allTop20.iloc[i]['artists'])
print()

print('minor all')
minorAll = data[['mode', 'popularity', 'track_name']]
minorAll = minorAll[minorAll['mode'] == 0]
print('minorAll head: ', minorAll.head(10))
minorAll = minorAll.sort_values(by=['popularity'], ascending=False)
print('minorAll head: ', minorAll.head(20))
print()

print('major all')
majorAll = data[['mode', 'popularity', 'track_name']]
majorAll = majorAll[majorAll['mode'] == 1]
print('majorAll head: ', majorAll.head(10))
majorAll = majorAll.sort_values(by=['popularity'], ascending=False)
print('majorAll head: ', majorAll.head(20))