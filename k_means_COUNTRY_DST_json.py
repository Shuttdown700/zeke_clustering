# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:17:37 2020

@author: shuttdown
"""
import collections
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
style.use('ggplot')

def clean(df, min_connections): # selects columns, drops N/As, more than 2 connections
    df = df[['IP_SRC','COUNTRY_DST','PORT_DST','IP_DST','COUNTRY_SRC','PORT_SRC','COUNT_BYTES_IN','COUNT_BYTES_OUT']]
    df = df.dropna().reset_index().drop(columns=['index'])
    df = df[df.PORT_DST<49152].reset_index().drop(columns=['index'])
    df = df[['IP_SRC','COUNTRY_DST','IP_DST','COUNTRY_SRC','PORT_SRC','PORT_DST','COUNT_BYTES_IN','COUNT_BYTES_OUT']]
    key_list = list(collections.Counter(df['IP_SRC']).keys())
    count_list = list(collections.Counter(df['IP_SRC']).values())
    count_df = pd.DataFrame({"IP_SRC":key_list,"Connections":count_list})
    count_df = count_df[count_df.Connections>=min_connections].reset_index().drop(columns=['index'])
    df = df[df.IP_SRC.isin(list(count_df['IP_SRC']))].reset_index().drop(columns=['index'])
    return df

def augment(bro_clean): # adds a country number column (for kmeans)
    bro_clean['COUNTRY_DST_NUM'] = bro_clean['COUNTRY_DST']
    countries = list(set(bro_clean['COUNTRY_DST'])) + ['nan']
    bro_clean['COUNTRY_DST_NUM'] = bro_clean['COUNTRY_DST'].apply(lambda x: countries.index(x))
    bro_clean = bro_clean[bro_clean.COUNTRY_DST_NUM!=len(countries)-1].reset_index().drop(columns=['index'])
    return bro_clean

def get_country_NUM_dictionary(df): # gets the country_num:country dictionary
    return dict(zip(list(set(df['COUNTRY_DST_NUM'])), set(df['COUNTRY_DST'])))

def normalize(inputVector): # normalized a feature vector
    norm_set = []
    for j in range(len(inputVector)):
        norm_set.append([inputVector[j][i] / sum(inputVector[j]) for i in range(len(inputVector[j]))])
    return np.array(norm_set)

def determine_optimal_k(measures_list, min_threshold = 0.05): # returns the optimal k value for the k-means
    opt_k_list = []
    for m in range(len(measures_list)):
        k = 1
        rang = max(measures_list[m]) - min(measures_list[m])
        diffs = []
        for i in range(1,len(measures_list[m])):
            diffs.append((measures_list[m][i-1] - measures_list[m][i])/rang)
        for i in range(len(diffs)-1):
            if diffs[i] < min_threshold and diffs[i+1] < min_threshold:
                break
            k += 1
        opt_k_list.append(k)
    # print(opt_k_list)
    return int(np.median(opt_k_list))

def elbow_methods(df, max_k): # uses the elbow method to determine optimal k value in k-means
    df_feature_vector = pd.crosstab(df.IP_SRC, df.COUNTRY_DST_NUM).reset_index()
    inputVector = np.array(df_feature_vector.iloc[:,1:])
    norm_inputVector = normalize(inputVector)
    X = norm_inputVector
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1,max_k)
    for k in K: 
        kmeanModel = KMeans(n_clusters=k, random_state = 12543, n_init=50).fit(X) 
        kmeanModel.fit(X)     
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kmeanModel.inertia_) 
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                     'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 
    distortions = []   
    for key,val in mapping1.items(): 
        distortions.append(val)
    inertias = []
    for key,val in mapping2.items(): 
        inertias.append(val)
    opt_k = determine_optimal_k([distortions,inertias])
    plt.plot(K,distortions, 'b')
    plt.xlabel('Values of K') 
    plt.ylabel('cDistances')
    plt.scatter(opt_k,distortions[opt_k-1], s=150, marker = '*')
    plt.title('The Elbow Method using Distortion') 
    plt.show()
    plt.plot(K, inertias, 'b') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia')
    plt.scatter(opt_k,inertias[opt_k-1], s=150, marker = '*')
    plt.title('The Elbow Method using Inertia') 
    plt.show()
    return opt_k

# def find_init_centers(num_clusters, df_feature_vector):
#     sum_column = list(df_feature_vector.sum())[1:]
#     cols = sorted(zip(sum_column, list(df_feature_vector.columns[1:])), reverse=True)[:num_clusters]
#     indices = [cols[i][1] for i in range(len(cols))]
#     centers = []
#     for i in range(len(indices)):
#         centroid_center = []
#         for j in range(len(sum_column)):
#             if j == indices[i]:
#                 centroid_center.append(1)
#             else:
#                 centroid_center.append(0)
#         centers.append(centroid_center)
#     return np.array(centers)

def get_kmeans(df, num_clusters): # k-means clustering for sets of feature vectors
    df = bro_aug
    df0 = df.groupby(['IP_SRC','COUNTRY_DST_NUM']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])
    df1 = df0.assign(normalized=df0.N.div(df0.IP_SRC.map(df0.groupby('IP_SRC').N.sum())))
    df_feature_vector = pd.crosstab(df.IP_SRC, df.COUNTRY_DST_NUM).reset_index()
    inputVector = np.array(df_feature_vector.iloc[:,1:])
    norm_inputVector = normalize(inputVector)
    kmeans = KMeans(n_clusters=num_clusters, random_state = 12543, n_init=50).fit(norm_inputVector)
    df_clusters = pd.DataFrame()
    df_clusters['IP_SRC'] = list(df_feature_vector['IP_SRC'])
    df_clusters['Cluster'] = list(kmeans.labels_)
    df2 = df1.merge(df_clusters, on='IP_SRC', how='inner')
    return df2, kmeans, norm_inputVector, df_feature_vector

def cluster_graphic(df, cluster, dictionary): # generates a graphic of a k-means cluster
    total_IPs = len(set(df['IP_SRC']))
    df_single_cluster = df[df.Cluster==cluster].reset_index().drop(columns=['index'])
    num_IPs = len(set(df_single_cluster['IP_SRC']))
    count = df_single_cluster.groupby(['COUNTRY_DST_NUM']).size()
    norm = [float(i)/sum(list(count)) for i in list(count)]
    xaxis = [dictionary[x] for x in list(count.keys())]
    plt.figure()
    plt.bar(xaxis,norm,color='Black')
    plt.xlabel('COUNTRY_DST')
    plt.ylabel('Connection % of Cluster')
    plt.title('Cluster {} :: {} Source IPs ({}% of total IPs)'.format(cluster+1,num_IPs,round(round(num_IPs/total_IPs,4)*100,2)))
    plt.show()

def all_cluster_graphics(df,kmeans,dictionary): # generates all k-means cluster graphics
    for c in range(len(kmeans.cluster_centers_)):
        cluster_graphic(df, c, dictionary)

def finding_outliers_local_global(df_feature_vector, norm_inputVector, kmeans, alpha): # finds outliers in clusters
    global_mean = np.mean(norm_inputVector, axis=0)
    df = pd.DataFrame()
    df['local_diff'] = [np.sum(norm_inputVector[i] - kmeans.cluster_centers_[kmeans.labels_[i]]) for i in range(len(norm_inputVector))]
    cluster_global_diffs = [np.sum(kmeans.cluster_centers_[i] - global_mean) for i in range(len(kmeans.cluster_centers_))]
    df['global_diff'] = [np.sum(norm_inputVector[i] - global_mean) for i in range(len(norm_inputVector))]
    df['IP_SRC'] = [df_feature_vector['IP_SRC'][i] for i in list(range(len(norm_inputVector)))]
    df['Cluster'] = [kmeans.labels_[i] for i in list(range(len(norm_inputVector)))]
    avg_local_diffs = list(df.groupby('Cluster').mean()['local_diff'])
    cluster_global_diffs = list(df.groupby('Cluster').mean()['global_diff'])
    # densities = [ list(collections.Counter(df['Cluster']).values())[i]/ sum(list(collections.Counter(df['Cluster']).values()))for i in range(len(list(collections.Counter(df['Cluster']).values())))]
    df['local_outlier_scores'] = [df['local_diff'][i] / avg_local_diffs[df['Cluster'][i]] for i in range(len(df))]
    df['global_outlier_scores'] = [(df['global_diff'][i] / cluster_global_diffs[df['Cluster'][i]]) for i in range(len(df))]
    df['outlier_scores'] = [(df['local_outlier_scores'][i] * df['global_outlier_scores'][i]) for i in range(len(df['global_outlier_scores']))]
    df = df.dropna().reset_index().drop(columns=['index'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace([np.inf, -np.inf], max(abs(df['outlier_scores'])))
    std = np.std(list(df['outlier_scores']))
    mean = np.mean(list(df['outlier_scores']))
    df['Z'] = [(df['outlier_scores'][i] - mean)/std for i in range(len(df['outlier_scores']))]
    t_crit = abs(scipy.stats.t.ppf(alpha/2, len(df) - 1))
    bins = int((max(df['Z'])-min(df['Z']))*(len(df)**(1/3))/(3.49*np.std(df['Z'])))
    plt.figure()
    plt.hist(df['Z'], color='Red', bins=bins)
    plt.xlabel('Z-Score for Outlier Score')
    plt.ylabel('Frequency')
    plt.title('Z-Score Distribution')
    plt.axvline(t_crit,0,color="Green",linestyle='dashed')
    plt.axvline(-1*t_crit,color="Green",linestyle='dashed')
    plt.show()
    df_outliers = df[abs(df.Z)>t_crit].sort_values(by=['Z'],ascending=True).reset_index().drop(columns=['index','local_diff','global_diff'])
    df_outliers = df_outliers.reindex(df_outliers.Z.abs().sort_values(ascending=False).index).reset_index().drop(columns=['index'])
    print('{} Source IPs identified as outliers with {}% confidence:'.format(len(set(df_outliers['IP_SRC'])),round(((1-alpha)*100),6)))
    for i in range(len(df_outliers)):
        print(df_outliers['IP_SRC'][i])
    return df, df_outliers

def plot_outliers(outliers, df_feature_vector, norm_inputVector, dictionary, max_plots): # plots traffic for outlier IP_SRC
    inputVector = np.array(df_feature_vector.iloc[:,1:])
    outliers = outliers.reindex(outliers.Z.abs().sort_values(ascending=False).index).reset_index().drop(columns=['index'])
    capped_outliers = pd.DataFrame()
    ranks = []
    index = 0
    pairings = []
    while len(capped_outliers) < max_plots and index < len(outliers)-1:
        if [outliers.loc[index,'Cluster'],outliers.loc[index,'Z'],outliers.loc[index,'local_outlier_scores']] not in pairings:
            capped_outliers = capped_outliers.append(outliers.iloc[index])
            pairings.append([outliers.loc[index,'Cluster'],outliers.loc[index,'Z'],outliers.loc[index,'local_outlier_scores']])
            ranks.append(index+1)
        index += 1
    capped_outliers = capped_outliers.reset_index().drop(columns=['index'])
    indices = [list(df_feature_vector['IP_SRC']).index(capped_outliers['IP_SRC'][i]) for i in range(len(capped_outliers))]
    feature_vectors = [list(list(norm_inputVector)[indices[i]]) for i in range(len(indices))]
    for f in range(len(feature_vectors)):
        xaxis = list(range(len(feature_vectors[f])))
        xaxis = [dictionary[x] for x in xaxis]
        x = []
        y = []
        for i in range(len(feature_vectors[f])):
            if feature_vectors[f][i] > 0:
                x.append(xaxis[i])
                y.append(feature_vectors[f][i])
        plt.figure()
        plt.bar(x,y,color='Green')
        plt.xlabel('COUNTRY_DST')
        plt.ylabel('Connection % of IP_SRC')
        plt.title('Outlier {}: {} ({} connections, cluster {})'.format(ranks[f],df_feature_vector['IP_SRC'][indices[f]],sum(inputVector[indices[f]]),pairings[f][0]+1))
        plt.show()

def build_analytic_table(df_scores, norm_inputVector):
    df = pd.DataFrame()
    df['IP_SRC'] = df_scores['IP_SRC']
    df['Outlier Score'] = df_scores['outlier_scores']
    df['Local Score'] = df_scores['local_outlier_scores']
    df['Global Score'] = df_scores['global_outlier_scores']
    columns = [countryNUM_dictionary[i] for i in range(len(norm_inputVector[0]))]
    df_norm_inputVector=pd.DataFrame(norm_inputVector, columns=columns)
    df = df.join(df_norm_inputVector)
    return df

def collect_data(bro_aug):
    print('Collecting Data...')
    df0 = bro_aug.groupby(['IP_SRC','IP_DST']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])
    df1 = bro_aug.groupby(['IP_SRC','COUNTRY_DST']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])    
    df2 = bro_aug.groupby(['IP_SRC','COUNTRY_SRC']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])    
    df3 = bro_aug.groupby(['IP_SRC','PORT_SRC']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])    
    df3['PORT_SRC'] = [str(int(df3['PORT_SRC'][i])) for i in range(len(list(df3['PORT_SRC'])))]
    df4 = bro_aug.groupby(['IP_SRC','PORT_DST']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])
    df4['PORT_DST'] = [str(int(df4['PORT_DST'][i])) for i in range(len(list(df4['PORT_DST'])))]
    df5 = bro_aug.groupby(['IP_SRC','COUNT_BYTES_IN']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])
    df5['COUNT_BYTES_IN'] = [str(int(df5['COUNT_BYTES_IN'][i])) for i in range(len(list(df5['COUNT_BYTES_IN'])))]    
    df6 = bro_aug.groupby(['IP_SRC','COUNT_BYTES_OUT']).size().reset_index().rename(columns={0:'N'}).sort_values('N', ascending=False).reset_index().drop(columns=['index'])
    df6['COUNT_BYTES_OUT'] = [str(int(df6['COUNT_BYTES_OUT'][i])) for i in range(len(list(df6['COUNT_BYTES_OUT'])))]  
    ip_srcs = list(set(list(df0['IP_SRC'])))
    ip_dests = []
    ip_countries_dst = []
    ip_countries_src = []
    ip_ports_dst = []
    ip_ports_src = []
    ip_bytes_in = []
    ip_bytes_out = []
    for s in range(len(ip_srcs)):
        dest_indices = [i for i, x in enumerate(list(df0['IP_SRC'])) if x == ip_srcs[s]]
        ip_dests.append([{list(df0['IP_DST'])[dest_indices[d]]:list(df0['N'])[dest_indices[d]]} for d in range(len(dest_indices))])
        src_indices = [i for i, x in enumerate(list(df1['IP_SRC'])) if x == ip_srcs[s]]
        ip_countries_dst.append([{list(df1['COUNTRY_DST'])[src_indices[d]]:list(df1['N'])[src_indices[d]]} for d in range(len(src_indices))])
        country_src_indices = [i for i, x in enumerate(list(df2['IP_SRC'])) if x == ip_srcs[s]]
        ip_countries_src.append([{list(df2['COUNTRY_SRC'])[country_src_indices[d]]:list(df2['N'])[country_src_indices[d]]} for d in range(len(country_src_indices))])
        if ip_countries_src[-1] == []:
            ip_countries_src[-1] = 'NA'
        port_dst_indices = [i for i, x in enumerate(list(df4['IP_SRC'])) if x == ip_srcs[s]]
        ip_ports_dst.append([{list(df4['PORT_DST'])[port_dst_indices[d]]:list(df4['N'])[port_dst_indices[d]]} for d in range(len(port_dst_indices))])
        src_dst_indices = [i for i, x in enumerate(list(df3['IP_SRC'])) if x == ip_srcs[s]]
        ip_ports_src.append([{list(df3['PORT_SRC'])[src_dst_indices[d]]:list(df3['N'])[src_dst_indices[d]]} for d in range(len(src_dst_indices))])    
        bytes_in_indices = [i for i, x in enumerate(list(df5['IP_SRC'])) if x == ip_srcs[s]]
        ip_bytes_in.append([{list(df5['COUNT_BYTES_IN'])[bytes_in_indices[d]]:list(df5['N'])[bytes_in_indices[d]]} for d in range(len(bytes_in_indices))])    
        bytes_out_indices = [i for i, x in enumerate(list(df6['IP_SRC'])) if x == ip_srcs[s]]
        ip_bytes_out.append([{list(df6['COUNT_BYTES_OUT'])[bytes_out_indices[d]]:list(df6['N'])[bytes_out_indices[d]]} for d in range(len(bytes_out_indices))])    
    collected_data = {str(ip_srcs[i]): {"IP_DST":ip_dests[i],
                                        "COUNTRY_DST":ip_countries_dst[i],
                                        "COUNTRY_SRC":ip_countries_src[i],
                                        "PORT_SRC":ip_ports_src[i],
                                        "PORT_DST":ip_ports_dst[i],
                                        "COUNT_BYTES_IN":ip_bytes_in[i],
                                        "COUNT_BYTES_OUT":ip_bytes_out[i]} for i in range(len(ip_dests))}
    return collected_data


def convert_to_json(df_scores,collected_data):
    anomaly_nums = list(range(1,len(df_scores)+1))
    max_length = len(str(len(anomaly_nums)))
    anomaly_nums = ['A'+str(anomaly_nums[i]) for i in range(len(anomaly_nums))]
    for i in range(len(anomaly_nums)):
        while len(anomaly_nums[i][1:]) < max_length:
            anomaly_nums[i] = anomaly_nums[i][0] + "0" + anomaly_nums[i][1:]
    ip_src = df_scores['IP_SRC']
    outlier_scores = df_scores['outlier_scores']
    z_scores = df_scores['Z']
    data = {}
    for a in range(len(anomaly_nums)):
        data.update({str(anomaly_nums[a]):{
            "SRC_IP":ip_src[a],
            "DST_IP":collected_data[ip_src[a]]['IP_DST'],
            "detection_type":"kmeans_COUNTRY_DST",
            "detection_timestamp": time.time(),
            "outlier_score": outlier_scores[a],
            "zscore":z_scores[a],
            "src_data":"bro",
            "src_port":collected_data[ip_src[a]]['PORT_SRC'],
            "dst_port":collected_data[ip_src[a]]['PORT_DST'],
            "traffic_type":'NA',
            "src_loc":collected_data[ip_src[a]]['COUNTRY_SRC'],
            "dst_loc":collected_data[ip_src[a]]['COUNTRY_DST'],
            "bytes_out":collected_data[ip_src[a]]['COUNT_BYTES_OUT'],
            "bytes_in":collected_data[ip_src[a]]['COUNT_BYTES_IN']}})
    return data

### read, clean, augment data ###
os.chdir("C:/Users/shuttdown/Documents/West Point/Firstie Year/SE402 - Capstone")
bro = pd.read_csv('bro_data.csv', low_memory=False)
bro_clean = clean(bro, min_connections=5)
bro_aug = augment(bro_clean)
# bro_clean.to_csv('bro_data_clean.csv')
# bro_aug.to_csv('bro_data_aug.csv')

### get dictionary ###
countryNUM_dictionary = get_country_NUM_dictionary(bro_aug)

### Use the elbow method to determine how many k clusters we need ###
max_k = 10
num_clusters = elbow_methods(bro_aug, max_k)

### K-means clustering by COUNTRY_DST ###
df_kmeans, kmeans, norm_inputVector, df_feature_vector = get_kmeans(bro_aug, num_clusters)
# all_cluster_graphics(df_kmeans,kmeans, countryNUM_dictionary)

### Identifying outliers ###
df_scores, df_outliers = finding_outliers_local_global(df_feature_vector, norm_inputVector, kmeans, alpha = 0.001)

### Return analytic table ###
df_analytic = build_analytic_table(df_scores, norm_inputVector)

### Plots outliers country distributions ###
plot_outliers(df_outliers, df_feature_vector, norm_inputVector, countryNUM_dictionary, max_plots = 10)

### Covert to Json ###
collected_data = collect_data(bro_aug)
json_data = convert_to_json(df_scores,collected_data)
json_string = str(json_data)
json_string = json_string.replace("'", '"')
f = open("demo_json.txt", "a")
f.write(json_string)
f.close()
############# Workspace #######################################################




