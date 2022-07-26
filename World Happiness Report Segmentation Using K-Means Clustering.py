#!/usr/bin/env python
# coding: utf-8

# # TASK #1 UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE
# 

# ![image.png](attachment:image.png)

# # TASK #2: IMPORT DATASETS AND LIBRARIES

# In[1]:


# Get the required dependencies
get_ipython().system(' pip install bubbly')
get_ipython().system(' pip install iplot')
get_ipython().system(' pip install chart_studio')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from chart_studio.plotly import plot, iplot
from plotly.offline import iplot

import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Import csv file into pandas dataframe
happy_df= pd.read_csv('happiness_report.csv')


# In[4]:


# print the first 5 rows of the dataframe
happy_df


# MINI CHALLENGE #1: 
# - Find out how many samples exist in the DataFrame using two different methods.
# - Select your own country from the dataframe and explore scores. Perform sanity check. 

# In[5]:


happy_df.shape


# In[6]:


happy_df[happy_df['Country or region'] == 'India']


# # TASK #3: PERFORM EXPLORATORY DATA ANALYSIS

# In[7]:


# Check the number of non-null values in the dataframe
happy_df.info()


# In[8]:


# Check Null values
happy_df.isnull().sum()


# In[9]:


# Obtain the Statistical summary of the dataframe
happy_df.describe()


# In[10]:


# check the number of duplicated entries in the dataframe
# since there are no duplicates, no further action is required
happy_df.duplicated().sum()


# MINI CHALLENGE #2:
# - What is the country that has the maximum happiness score? What is the perception of corruption in this country?

# In[11]:


country_with_max_score = happy_df[happy_df['Score'] == happy_df['Score'].max()]['Country or region'].iloc[0]
perception_of_corruption = happy_df[happy_df['Score'] == happy_df['Score'].max()]['Perceptions of corruption'].iloc[0]
print('Country with maximum happiness score : ',country_with_max_score)
print(f'{country_with_max_score}\'s perception of corruption : ',perception_of_corruption)


# # TASK #4: PERFORM DATA VISUALIZATION - PART #1

# In[12]:


# Plot the pairplot
fig = plt.figure(figsize = (20,20))

# Positive correlation between GDP and score 
# Positive correlation between Social Support and score 
sns.pairplot(happy_df.drop(['Overall rank','Country or region'],axis=1))


# In[13]:


# distplot combines the matplotlib.hist function with seaborn kdeplot()
columns = ['Score','GDP per capita', 'Social support', 'Healthy life expectancy', 
    'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']
plt.figure(figsize = (20, 50))
for i in range(len(columns)):
  plt.subplot(4, 2, i+1)
  sns.distplot(happy_df[columns[i]], color = 'r');
  plt.title(columns[i])

plt.tight_layout()


# MINI CHALLENGE #3:
# - Plot the correlation matrix and comment on the results. 
# 

# In[14]:


plt.figure(figsize=(20,20))
plt.rcParams.update({'font.size':15})
sns.heatmap(happy_df[columns].corr(),annot=True)


# # TASK #5: PERFORM DATA VISUALIZATION - PART #2

# In[15]:


# Plot the relationship between score, GDP and region
fig = px.scatter(happy_df,x='GDP per capita',y='Score',color='Score',size='Score',hover_data = happy_df.columns)
fig.update_layout(title_text = 'GDP per Capita vs Happiness Score')
fig.show()


# In[16]:


# Plot the relationship between score and freedom to make life choices

fig = px.scatter(happy_df, x = 'Freedom to make life choices', y = "Score", size = 'Score', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")

fig.update_layout(title_text = 'Happiness Score vs Freedom to make life choices')
fig.show()


# MINI CHALLENGE #4: 
# - Plot a similar type of plots for 'Healthy life expectancy' and 'Score'

# In[17]:


fig = px.scatter(happy_df, x = 'Healthy life expectancy', y = "Score", size = 'Score', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")

fig.update_layout(title_text = 'Happiness Score vs Healthy life expectancy')
fig.show()


# # TASK #6: PREPARE THE DATA TO FEED THE CLUSTERING MODEL

# In[18]:


# We are going to create clusters without the use of happiness score and rank to see which countries fall under similar clusters


# In[19]:


# Select the data without rank and happiness score
df_seg = happy_df.drop(['Overall rank','Country or region','Score'],axis=1)


# In[20]:


df_seg.sample(frac=0.05)


# In[21]:


# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_seg)


# In[22]:


scaled_data.shape


# # TASK #7: UNDERSTAND THE THEORY AND INTUITION BEHIND K-MEANS CLUSTERING ALGORITHM

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #8: FIND THE OPTIMAL NUMBER OF CLUSTERS USING ELBOW METHOD

# - The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset. 
# - If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.
# - Source: 
#   - https://en.wikipedia.org/wiki/Elbow_method_(clustering)
#   - https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

# In[23]:


scores = []
for k in range(1,20):
    kmean = KMeans(n_clusters=k)
    kmean.fit(scaled_data)
    scores.append(kmean.inertia_)

plt.plot(scores,'o-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores') 
plt.show()


# In[24]:


# From this we can observe that 3rd cluster seems to be forming the elbow of the curve. 
# Let's choose the number of clusters to be 3.


# # TASK #9: APPLY K-MEANS METHOD

# In[25]:


kmean = KMeans(n_clusters=3)


# In[26]:


kmean.fit(scaled_data)


# In[27]:


labels = kmean.labels_
labels


# In[28]:


cluster_centers = kmean.cluster_centers_
print(cluster_centers)


# In[29]:


cluster_centers = pd.DataFrame(data = kmean.cluster_centers_, columns = [df_seg.columns])
cluster_centers      


# In[30]:


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df_seg.columns])
cluster_centers


# - Cluster 0: countries that have GDP in the range of 0.6 to 1.4 and have high social support. These countries have medium life expectancy and have high freedom to make life choices. These counties have low generosity and low perception of corruption.
# 
# - Cluster 1: countries that have very high GDP, high social support and high life expectancy. These counties have high freedom to make life choices, medium generosity and medium perception of corruption.
# 
# - Cluster 2: countries that have low GDP average life expectancy and average social support. These counties have low freedom to make life choices, high generosity and medium perception of corruption.

# In[31]:


labels.shape # Labels associated to each data point


# In[32]:


labels.max()


# In[33]:


labels.min()


# In[34]:


y_kmeans = kmean.fit_predict(scaled_data)
y_kmeans


# In[35]:


# concatenate the clusters labels to our original dataframe
happy_df_cluster = pd.concat([happy_df, pd.DataFrame({'cluster':labels})], axis = 1)
happy_df_cluster


# In[36]:


# Plot the histogram of various clusters
for i in df_seg.columns:
  plt.figure(figsize = (35, 10))
  for j in range(3):
    plt.subplot(1,3,j+1)
    cluster = happy_df_cluster[happy_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i, j))
  
  plt.show()


# MINI CHALLENGE #5:
# - Try the same model with 4 clusters 

# In[37]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_data)
col_labels=kmeans.labels_
cluster_center=kmeans.cluster_centers_
cluster_center = scaler.inverse_transform(cluster_center)
cluster_center = pd.DataFrame(data = cluster_center, columns = [df_seg.columns])
cluster_center


# In[38]:


happy_df_cluster_new = pd.concat([happy_df, pd.DataFrame({'cluster':col_labels})], axis = 1)
happy_df_cluster_new


# In[39]:


for i in df_seg.columns:
  plt.figure(figsize = (35, 10))
  for j in range(4):
    plt.subplot(1,4,j+1)
    cluster = happy_df_cluster_new[happy_df_cluster_new['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i, j))
  
  plt.show()


# # TASK #10: VISUALIZE THE CLUSTERS

# In[40]:


happy_df_cluster


# In[41]:


# Plot the relationship between cluster and score 

fig = px.scatter(happy_df_cluster, x = 'cluster', y = "Score",
           size = 'Score', color = "Country or region", hover_data = happy_df.columns,hover_name='Country or region',
          trendline = "ols")

fig.update_layout(title_text = 'Cluster vs Happiness Score')
fig.show()


# In[42]:


# Plot the relationship between cluster and GDP

fig = px.scatter(happy_df_cluster, x = 'cluster', y = "GDP per capita",
           size = 'GDP per capita', color = "Country or region", hover_data = happy_df.columns,hover_name='Country or region',
          trendline = "ols")
fig.update_layout(title_text='Clusters vs GDP per capita')
fig.show()


# In[43]:


# Visaulizing the clusters with respect to economy, corruption, gdp, rank and their scores

from bubbly.bubbly import bubbleplot

figure = bubbleplot(dataset=happy_df_cluster, 
    x_column='GDP per capita', y_column='Perceptions of corruption', bubble_column='Country or region',  
    color_column='cluster', z_column='Healthy life expectancy', size_column='Score',
    x_title="GDP per capita", y_title="Corruption", z_title="Life Expectancy",
    title='Clusters based Impact of Economy, Corruption and Life expectancy on Happiness Scores of Nations',
    colorbar_title='Cluster', marker_opacity=1, colorscale='Portland',
    scale_bubble=0.8, height=650)

iplot(figure, config={'scrollzoom': True})


# MINI CHALLENGE #6:
# - Plot the similar type of visualization having 'Generosity' instead of 'Healthy life expectancy'

# In[44]:


figure = bubbleplot(dataset=happy_df_cluster, 
    x_column='GDP per capita', y_column='Perceptions of corruption', bubble_column='Country or region',  
    color_column='cluster', z_column='Generosity', size_column='Score',
    x_title="GDP per capita", y_title="Corruption", z_title="Generosity",
    title='Clusters based Impact of Economy, Corruption and Generosity on Happiness Scores of Nations',
    colorbar_title='Cluster', marker_opacity=1, colorscale='Portland',
    scale_bubble=0.8, height=650)

iplot(figure, config={'scrollzoom': True})


# In[45]:


# Visualizing the clusters geographically
data = dict(type = 'choropleth', 
           locations = happy_df_cluster["Country or region"],
           locationmode = 'country names',
           colorscale='RdYlGn',
           z = happy_df_cluster['cluster'], 
           text = happy_df_cluster["Country or region"],
           colorbar = {'title':'Clusters'})

layout = dict(title = 'Geographical Visualization of Clusters', 
              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)

