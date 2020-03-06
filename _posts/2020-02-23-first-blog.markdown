---
layout: post
title:  "Mean shift Clustering"
date:   2020-02-23 16:46:36 +0530
categories: MachineLearning
---
Mean shift is a non-parametric feature-space analysis technique for locating the maxima of a density function, a so-called mode-seeking algorithm.[1] Application domains include cluster analysis in computer vision and image processing.


```python
def mean_shift(data, radius=2.0):
    raw_shape = data.shape
    data = data.reshape((int(data.size/data.shape[-1]),data.shape[-1]))
    clusters = []
    visited = [True]*len(data)
    while(len(data[visited])!=0):
        cluster_centroid = data[visited][0]
        visited[np.where(data==cluster_centroid)[0][0]]=False
        cluster_frequency = np.zeros(len(data))
        while True:
            temp_data = []
            for j in range(len(data)):
                v = data[j]
                if np.linalg.norm(v - cluster_centroid) <= radius:
                    temp_data.append(v)
                    visited[j] = False
                    cluster_frequency[j] += 1
            old_centroid = cluster_centroid
            new_centroid = np.average(temp_data, axis=0)
            cluster_centroid = new_centroid
            if np.array_equal(new_centroid, old_centroid):
                break
        for cluster in clusters:
            if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= 0.5*radius:
                cluster['frequency'] = cluster['frequency'] + cluster_frequency
                break
        else:
            clusters.append({
                'centroid': cluster_centroid,
                'frequency': cluster_frequency
            })
    t = []
    Y = np.zeros(len(data))
    for cluster in clusters:
        cluster['data'] = []
        t.append(cluster['frequency'])
    t = np.array(t)
    for i in range(len(data)):
        column_frequency = t[:, i]
        cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
        Y[i] = cluster_index
    Y.reshape(raw_shape[:-1])
    return Y
```

The vectorized version is
```python
def mean_shift_vector(data, radius=2.0):
    raw_shape = data.shape
    data = data.reshape((int(data.size/data.shape[-1]),data.shape[-1]))
    clusters = np.empty([0,len(data[0])])
    cluster_frequencies = np.empty([0,len(data)])
    visited = np.array([True]*len(data))
    while(len(data[visited])!=0):
        
        cluster_centroid = data[visited][0]
        visited[np.where(data==cluster_centroid)[0][0]]=False
        cluster_frequency = np.zeros(len(data))
        while True:
            distance = np.linalg.norm(data-cluster_centroid[np.newaxis,:],axis = 1)
            region = (distance<=radius)
            cluster_frequency+=region
            visited=~(region+(~visited))
            old_centroid = cluster_centroid
            new_centroid = np.average(data[region], axis=0)
            cluster_centroid = new_centroid
            if np.array_equal(new_centroid, old_centroid):
                break
                
        if not clusters.any():
            clusters = cluster_centroid[np.newaxis,:]
        if not cluster_frequencies.any():
            cluster_frequencies = cluster_frequency[np.newaxis,:]
            continue
        distance = np.linalg.norm(clusters-cluster_centroid[np.newaxis,:],axis = 1)
        region = (distance<=0.5*radius)
        if region.any():
            cluster_frequencies[region[0]] += cluster_frequency
        else:
            clusters = np.append(clusters,cluster_centroid[np.newaxis,:],axis = 0)
            cluster_frequencies = np.append(cluster_frequencies,cluster_frequency[np.newaxis,:],axis = 0)
    Y = np.argmax(cluster_frequencies,axis = 0)
    Y.reshape(raw_shape[:-1])
    return Y
```


