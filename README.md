# CSO-Anomaly-Detection



## Introduction
Combined Sewer Overflow (CSO) events are a major reason for water pollution in open water bodies. Each time the sewer system is loaded with more storm- and wastewater than the connected sewer treatment plant can handle, the overload of storm- and wastewater is pumped into adjacent water bodies \cite{bwb}. Therefore it is necessary to detect possible incoming CSO events, in order to prevent an overload of the sewage system and be able to initiate counter measures before pumping additional wastewater into nature.  
This project aims to create an anoamly detection framework for preventing the emission of polluted water into adjacent ecosystems. Suitable data from weather stations are flow rate (FR) and level measurement (LM), as well as the amount of waste water in the canalization during the same time. Fluctuations of waste water and precipitation are highly dependent on daily cycles, as well as the seasons. The importance of this temporal aspect will be caputred by a Long short-term memory (LSTM) network, wich is able to take shorter, aswell as longer time periods into account. The trained model will then be used to predict CSO events from unseen weather forecasts.  
## CSO Event vs. Non CSO Event
A typical day without CSO events. The level measurement of the groundwater is higher in the morning and evening due to citiziens showering, cooking etc. However this spikes are to be expected and do not lead to CSO events. 
<img src="images/nonCSOEvent.png">

Heavy rain fall lets the level measurment rise, which leads to a CSO event around noon. This leads to wasterwater being pumped into nearby rivers in order to relieve the sewers - and the level measurement sinks again.
<img src="images/anomalousday.png">
## Architecture
## Performance

<img src="images/roc3m.png"><img src="images/MAE.png">


## Results
