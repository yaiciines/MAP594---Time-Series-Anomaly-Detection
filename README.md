# STUDY OF DENOISING DIFFUSION PROBABILISTIC MODELS FOR TIME SERIES ANOMALY DETECTION
MAP594 - Modélisation probabiliste et statistique - Time Series Anomaly Detection 

Anomaly detection in time series data has a great importance to ensure efficiency in industrial
contexts across various domains. However, the accurate detection of anomalies in such data
poses significant challenges due to the unexpected behaviors and the need for accurate modeling
of real time series data, in addition to the difficulty of labeling the data.
The most commonly used theoretical methods are generative adversarial networks or auto-
encoders, or existing deep learning approaches, including predictive and reconstruction-based
methods, models struggle in this context to address these challenges effectively as they are time-
consuming and require computational and human resources, which can be costly for companies
in the industrial sector. On the other hand, well-known statistical methods such as SVM or
Isolation Forest, which are fast and do not require much memory, do not explicitly handle
temporal information.
To address these challenges, I present an unsupervised anomaly detection method based on
denoising diffusion models, which have gained recognition in recenhttps://www.google.com/t years for their impressive
performance in generative modeling and are promising candidates for density-based anomaly
detection due to their ability to capture complex data distributions. However, despite their
potential, these models have not yet been widely applied to anomaly detection, making this
approach both innovative and exploratory in the field. The implementation involves training
the noise and denoising scheme of DDPMs with different types of noise, integrated with a
Temporal Convolutional Network (TCN) to effectively detect and process time sequences.


# Datasets 

The datasets used in this study are selected from those identified by Si et al [1] which include
well-known time series datasets and benchmark performances of various anomaly detection
models. These datasets serve as a reference to facilitate easy comparison of the results found
with existing models. Since most of the relevant work in this area is recent (post-2020), it’s
crucial to carefully consider the information provided and its application.

# References 
[1] Haotian Si, Changhua Pei, Hang Cui, Jingwen Yang, Yongqian Sun, Shenglin Zhang,
Jingjing Li, Haiming Zhang, Jing Han, Dan Pei, Jianhui Li, and Gaogang Xie. Timeseries-
bench: An industrial-grade benchmark for time series anomaly detection models. arXiv
preprint arXiv:2402.10802, v2, 2024. arXiv:2402.10802v2 [cs.LG], 26 Feb 2024.

[2] Nab dataset. https://www.kaggle.com/datasets/boltzmannbrain/nab, 2017.

[3] Aiops dataset. https://github.com/iopsai/iops, 2018.

[4] Ucr time series anomaly archive. https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ucr-time-series-anomaly-archive, 2022.

[5] Wsd dataset. https://github.com/alumik/AnoTransfer-data, 2022.7[2]Yahoo webscope s5 dataset. https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70, 2015.

[6] Dataflowr. Project tiny diffusion. https://github.com/dataflowr/Project-tiny-diffusion.

