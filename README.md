# CSO-Anomaly-Detection



## Introduction
Combined Sewer Overflow (CSO) events are a major reason for water pollution in open water bodies. Each time the sewer system is loaded with more storm- and wastewater than the connected sewer treatment plant can handle, the overload of storm- and wastewater is pumped into adjacent water bodies \cite{bwb}. Therefore it is necessary to detect possible incoming CSO events, in order to prevent an overload of the sewage system and be able to initiate counter measures before pumping additional wastewater into nature.  
This project aims to create an anoamly detection framework for preventing the emission of polluted water into adjacent ecosystems. Suitable data from weather stations are flow precipitation and level of precipitation, as well as the amount of waste water in the canalization during the same time. Fluctuations of waste water and precipitation are highly dependent on daily cycles, as well as the seasons. The importance of this temporal aspect will be caputred by a Long short-term memory (LSTM) network, wich is able to take shorter, aswell as longer time periods into account. 
## Results
<img src="images/nonCSOEvent.png">
