"""
@author: hanwenzhang
"""
# imports. so many
import numpy as np
import pandas as pd
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, silhouette_samples, accuracy_score, roc_auc_score
#import xicorpy
#from xicor.xicor import Xi
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit

#loading in data
data = pd.read_csv("spotify52kData.csv")
data = data.dropna()
print(data.shape)

################################################
#  question 1: 10 song features distributions  #
################################################
selected_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

selected_data = data[selected_features]
# Plot histograms for each feature
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    sns.histplot(selected_data[feature], kde=True, ax=axes[i], bins=47)
    # Calculate skewness and kurtosis
    skewness = selected_data[feature].skew()
    kurt = selected_data[feature].kurtosis()    
    # Add caption with skewness and kurtosis
    axes[i].text(0.5, -0.5, f"Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}", 
                 transform=axes[i].transAxes, ha='center', va='center', fontsize=8)
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()


#########################################################
#  question 2: song length and popularity relationship  #
#########################################################
popularity = data['popularity']
duration = data['duration']
plt.figure(figsize=(10, 6))
plt.scatter(duration, popularity, alpha=0.5, color='blue')
plt.title('Song Length vs Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.grid(True)

pearson_corr = duration.corr(popularity)
print('pearson_corr: ', pearson_corr)
spearman_corr = duration.corr(popularity, method='spearman')
print('spearman_corr: ', spearman_corr)

'''
#chatterjee stuff. did not work :)
popNP = popularity.tolist()
durNP = duration.tolist()
#xi = xicorpy.compute_xi_correlation(popNP, durNP)
xi_obj = Xi(popNP, durNP)
print('after xi_obj')
print('xi_obj.correlation: ', xi_obj.correlation)
#\chatterjee_corr = xi_obj.correlation
#print('chatterjee_corr: ', chatterjee_corr)
#pvals = xi_obj.pval_asymptotic(ties=False, nperm=1000)
#print('chatterjee_corr: ', chatterjee_corr)
'''

plt.text(0.5, -0.13,f"Pearson Correlation Coefficient: {pearson_corr:.2f}   Spearman Correlation Coefficient: {spearman_corr:.2f}", 
         transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)

plt.ticklabel_format(style='plain', axis='x')
plt.show()


#########################################################################
#  question 3: popularity: explicit vs non-explicit, significance test  #
#########################################################################
sns.histplot(popularity, kde=True, bins=47).set_title('Popularity Distribution')
skewness = popularity.skew()
kurt = popularity.kurtosis()    

plt.tight_layout()
plt.show()
print('skewness: ', skewness)
print('kurtosis: ', kurt)

allByExplicit = data[['explicit', 'popularity']]
explicit = allByExplicit[allByExplicit['explicit'] == True]
print('Explicit Median: ', explicit['popularity'].median())
print('Explicit Count: ', explicit.shape[0])
sns.histplot(explicit['popularity'], kde=True, bins=47).set_title('Explicit Popularity Distribution')
plt.show()
nonExplicit = allByExplicit[allByExplicit['explicit'] == False]
print('Non-Explicit Median: ', nonExplicit['popularity'].median())
print('Non-Explicit Count: ', nonExplicit.shape[0])
sns.histplot(nonExplicit['popularity'], kde=True, bins=47).set_title('Non-Explicit Popularity Distribution')
plt.show()
print()

u_statistic, p_value = stats.mannwhitneyu(explicit['popularity'], nonExplicit['popularity'], alternative='greater')

# Plot a figure illustrating the findings
plt.figure(figsize=(10, 6))
sns.boxplot(x='explicit', y='popularity', data=allByExplicit)
plt.title('Popularity of Explicit vs Non-Explicit Songs')
plt.xlabel('Explicit')
plt.ylabel('Popularity')
plt.show()

# Check the significance level
alpha = 0.05
print(f"P-value: {p_value} < alpha: {alpha}")
print()
if p_value < alpha:
    print("The difference in popularity between explicit and non-explicit songs is statistically significant.")
    print("Explicit songs are, on average, more popular than non-explicit songs.")
else:
    print("There is no significant difference in popularity between explicit and non-explicit songs.")
print()


###############################################################
#  question 4: popularity: major vs minor, significance test  #
###############################################################
mode = data[['mode', 'popularity']]
major = mode[mode['mode'] == 1]
minor = mode[mode['mode'] == 0]
print('Major Key Median: ', major['popularity'].median())
print('Major Key Count: ', major.shape[0])
sns.histplot(major['popularity'], kde=True, bins=47).set_title('Major Key Popularity Distribution')
plt.show()
print()
print('Minor Key Median: ', minor['popularity'].median())
print('Minor Key Count: ', minor.shape[0])
sns.histplot(minor['popularity'], kde=True, bins=47).set_title('Minor Key Popularity Distribution')
plt.show()
print()

t_statistic, p_value = stats.ttest_ind(major['popularity'], minor['popularity'], equal_var=False)

# Plot a figure illustrating the findings
plt.figure(figsize=(10, 6))
sns.boxplot(x='mode', y='popularity', data=mode)
plt.title('Popularity of Major vs Minor Songs')
plt.xlabel('Mode')
plt.ylabel('Popularity')
plt.show()

alpha = 0.05
print(f"P-value: {p_value} < alpha: {alpha}")
print()
if p_value < alpha:
    print("The difference in popularity between major and minor songs is statistically significant.")
    if t_statistic > 0:
        print("Major key songs are, on average, more popular than minor songs.")
    else:
        print("Minor key songs are, on average, more popular than major key songs.")
else:
    print("There is no significant difference in popularity between major and minor key songs.")


#######################################
#  question 5: energy is loudness???  #
#######################################
energy = data['energy']
loudness = data['loudness']

plt.figure(figsize=(10, 6))
plt.scatter(energy, loudness, alpha=0.5, color='blue')
plt.title('Energy vs Loudness')
plt.xlabel('Energy')
plt.ylabel('Average Loudness, in db')
plt.grid(True)

pearson_corr = energy.corr(loudness)
print('pearson_corr: ', pearson_corr)
spearman_corr = energy.corr(loudness, method='spearman')
print('spearman_corr: ', spearman_corr)
plt.text(0.5, -0.13,f"Pearson Correlation Coefficient: {pearson_corr:.2f}   Spearman Correlation Coefficient: {spearman_corr:.2f}", 
         transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)

plt.ticklabel_format(style='plain', axis='x')
plt.show()


############################################################################
#  question 6:                                                             #
#   Which of the 10 song features in question 1 predicts popularity best?  #
#   How good is this model?                                                #
############################################################################
popularity = popularity.to_numpy()
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
axes = axes.flatten()

#predicting popularity with each feature
for i, feature in enumerate(selected_features):
    x_train, x_test, y_train, y_test = train_test_split(data[feature], popularity, test_size=0.2, random_state=1234567890)
    x_train = x_train.values.reshape(-1,1)
    x_test = x_test.values.reshape(-1,1)
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rSq = model.score(x_test, y_test)
    slope = model.coef_
    intercept = model.intercept_
    yHat = slope * popularity + intercept
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    axes[i].scatter(x_test, y_test, color='black')
    axes[i].plot(x_test, y_pred, color='orange', linewidth=3)
    title = 'R^2 = {:.3f}'.format(rSq) + '\n' + 'RMSE = {:.3f}'.format(rmse)
    axes[i].set_title(title)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('popularity')

plt.tight_layout()
plt.show()


#######################################################################################
#  question 7:                                                                        #
#   7) Building a model that uses all of the song features in question 1,             #
#   how well can you predict popularity?                                              #
#   How much (if at all) is this model improved compared to the model in question 6)  #
#   How do you account for this?                                                      #
#######################################################################################
tenFeatures = selected_data.to_numpy()
print('tenFeatures type', type(tenFeatures))
x_train, x_test, y_train, y_test = train_test_split(tenFeatures, popularity, test_size=0.2, random_state=1234567890)

model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
rSq = model.score(x_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
plt.scatter(y_pred, y_test, color='black')
plt.plot(y_test, y_test, color='orange', linewidth=3)
plt.title('Linear Regression model\n'+'R^2 = {:.3f}'.format(rSq) + '    ' + 'RMSE = {:.3f}'.format(rmse))
plt.xlabel('predicted popularity')
plt.ylabel('actual popularity')
plt.show()

model = Ridge(alpha=100).fit(x_train, y_train)
y_pred = model.predict(x_test)
rSq = model.score(x_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
plt.scatter(y_pred, y_test, color='black')
plt.plot(y_test, y_test, color='orange', linewidth=3)
plt.title('Ridge Regression model\n'+'R^2 = {:.3f}'.format(rSq) + '    ' + 'RMSE = {:.3f}'.format(rmse))
plt.xlabel('predicted popularity')
plt.ylabel('actual popularity')
plt.show()


#################################################################################
#  question 8: When considering the 10 song features above,                     #
#   how many meaningful principal components can you extract?                   #
#   What proportion of the variance do these principal components account for?  #
#   Using the principal components, how many clusters can you identify?         #
#################################################################################
corrMatrix = np.corrcoef(tenFeatures,rowvar=False)
# Plot the data:
plt.imshow(corrMatrix) 
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()
# 2-3 features are highly correlated with each other.
# 1-8 features are highly correlated with each other.
# 2-5 features are negatively correlated with each other.

#PCA
zscoredData = stats.zscore(tenFeatures)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
varExplained = eigVals/sum(eigVals)*100
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

# What a scree plot is: A bar graph of the sorted Eigenvalues
numFeatures = 10
x = np.linspace(1,numFeatures,numFeatures)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numFeatures],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Scree plot')
plt.show()

whichPrincipalComponent = 3 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Features')
plt.ylabel('Loading')
plt.title('Loading plot for PC' + str(whichPrincipalComponent))
plt.show() # Show bar plot
print(selected_features) # Display questions to remind us what they were

plt.plot(rotatedData[:,0]*-1,rotatedData[:,1]*-1,'o',markersize=5) #Again the -1 is for polarity
plt.xlabel('upbeat')
plt.ylabel('polished')
plt.title('PCA: upbeat vs polished')
plt.show()

plt.plot(rotatedData[:,0]*-1,rotatedData[:,2]*-1,'o',markersize=5) #Again the -1 is for polarity
plt.xlabel('upbeat')
plt.ylabel('short + sweet')
plt.title('PCA: upbeat vs short&sweet')
plt.show()

plt.plot(rotatedData[:,1]*-1,rotatedData[:,2]*-1,'o',markersize=5) #Again the -1 is for polarity
plt.xlabel('polished')
plt.ylabel('short + sweet')
plt.title('PCA: polished vs short&sweet')
plt.show()

pca1 = rotatedData[:,0]*-1
pca2 = rotatedData[:,1]*-1
pca3 = rotatedData[:,2]*-1
x = np.column_stack((pca1, pca2, pca3))

# Init:
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters,1])*np.NaN # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii), random_state=1234567890).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

# kMeans:
numClusters = 2
kMeans = KMeans(n_clusters = numClusters, random_state=1234567890).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=1)
    #plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('upbeat')
    plt.ylabel('musical')
plt.show()

for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,2],'o',markersize=1)
    #plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),2],'o',markersize=5,color='black')  
    plt.xlabel('upbeat')
    plt.ylabel('short + sweet')
plt.show()

for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,1],x[plotIndex,2],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('musical')
    plt.ylabel('short + sweet')
plt.show()

print('rotatedData head: ', rotatedData[0:5,:])


#######################################################################################
#  question 9: Can you predict whether a song is in major or minor key from valence?  #
#   If so, how good is this prediction?                                               #
#   If not, is there a better predictor?                                              #
#   [Suggestion: It might be nice to show the logistic regression                     #
#   once you are done building the model]                                             #
#######################################################################################
predictingMode = data[['valence', 'mode']].to_numpy()
print(predictingMode[:10])
print()
# major song descriptives:
numMajor = len(np.argwhere(predictingMode[:,1]==1))
majorPredAvg = np.mean(predictingMode[np.argwhere(predictingMode[:,1]==1),0])
majorPredStd = np.std(predictingMode[np.argwhere(predictingMode[:,1]==1),0])
print('Number of major songs:',numMajor)
print('Major avg valence:',majorPredAvg.round(3))
print('Major valence std:',majorPredStd.round(3))
print()

# minor songs descriptives:
numMinor = len(np.argwhere(predictingMode[:,1]==0))
minorPredAvg = np.mean(predictingMode[np.argwhere(predictingMode[:,1]==0),0])
minorPredStd = np.std(predictingMode[np.argwhere(predictingMode[:,1]==0),0])
print('Number of minor songs:',numMinor)
print('Minor avg valence:',minorPredAvg.round(3))
print('Minor valence std:',minorPredStd.round(3))
print()

#making sure there's no missing data
#print('major+minor:',numMajor+numMinor)
#print('length of valence:',len(majorMinor[:,0]))

# Plot data:
plt.scatter(predictingMode[:,0],predictingMode[:,1],color='black')
plt.xlabel('valence')
plt.ylabel('major?')
plt.yticks(np.array([0,1]))
plt.show()

# Format data:
x = predictingMode[:,0].reshape(len(predictingMode),1) 
y = predictingMode[:,1]

# Fit model:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234567890)
model = LogisticRegression().fit(x_train, y_train)

x1 = np.linspace(0,1,100)
y1 = x1 * model.coef_ + model.intercept_
sigmoid = expit(y1)

plt.title('Logistic regression model')
plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3) # the ravel function returns a flattened array
plt.scatter(predictingMode[:,0],predictingMode[:,1],color='black')
plt.hlines(0.5,0,1,colors='gray',linestyles='dotted')
plt.xlabel('valence')
plt.xlim([0,1])
plt.ylabel('major?')
plt.yticks(np.array([0,1]))
plt.show()

y_pred = model.predict(x_test)
print('ypred:',y_pred)
print('ytest:',y_test)
confusionMatrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(confusionMatrix)

true_neg = confusionMatrix[0,0]
true_pos = confusionMatrix[1,1]
false_neg = confusionMatrix[1,0]
false_pos = confusionMatrix[0,1]

# Calculate accuracy:
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
print('Accuracy:',accuracy.round(3))
precision = true_pos / (true_pos + false_pos)
print('Precision:',precision.round(3))
recall = true_pos / (true_pos + false_neg)
print('Recall:',recall.round(3))
specifity = true_neg / (true_neg + false_pos)
print('Specifity:',specifity.round(3))

roc_auc = roc_auc_score(y_test, y_pred)
print('predicting popularity from valence AUC:',roc_auc.round(3))

########################################################################################
#  question 10: Can you predict the genre, either from the 10 song features            #
#   from question 1 directly or the principal components you extracted in question 8?  #
#   [Suggestion: Use a classification tree, but you might have to map the qualitative  #
#   genre labels to numerical labels first]                                            #  
########################################################################################  
genreLabels = data['track_genre'].unique()
print(genreLabels)
print('genreLabels length: ', len(genreLabels))

#PCA
x = np.column_stack((pca1, pca2, pca3))

print('pca1 range: ', min(pca1), ' ', max(pca1))

print('x head: ', x[0:5,:])
print('x shape: ', x.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['track_genre'])
#de_encoded_y = label_encoder.inverse_transform(y_encoded)

#with pca components
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=1234567890)
numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees, random_state=1234567890).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelAccuracy = accuracy_score(y_pred, y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print('Random forest model w PCA accuracy:',modelAccuracy)
print('Random forest model w PCA AUC:',roc_auc)
#classification tree
clf = DecisionTreeClassifier(random_state=1234567890).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelAccuracy = accuracy_score(y_pred, y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print('Classification tree w PCA accuracy:',modelAccuracy)
print('Classification tree w PCA AUC:',roc_auc)

#with all features
x_train, x_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=1234567890)
clf = RandomForestClassifier(n_estimators=numTrees, random_state=1234567890).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelAccuracy = accuracy_score(y_pred, y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print('Random forest model w all features accuracy:',modelAccuracy)
print('Random forest model w all features AUC:',roc_auc)
#classification trees
clf = DecisionTreeClassifier(random_state=1234567890).fit(x_train, y_train)
y_pred = clf.predict(x_test)
modelAccuracy = accuracy_score(y_pred, y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print('Classification tree w all features accuracy:',modelAccuracy)
print('Classification tree w all features AUC:',roc_auc)