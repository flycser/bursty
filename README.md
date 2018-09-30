# Bursty Repository

## Introduction 

This project is implementations of the burst detection model, which was firstly proposed by Jon Kleinberg in 2002 in the paper
> Kleinberg, Jon. "Bursty and hierarchical structure in streams." Data Mining and Knowledge Discovery 7.4 (2003): 373-397.

You can find more details about the paper and experiments from <https://www.cs.cornell.edu/home/kleinber/kdd02.html> 

Burst detection model identifies time periods (**bursts**) in a document streams. A burst indicates the appearance of a topic, which emerges accompanied with certain features rising sharply in frequency.   

The approach is based on modeling the stream using an **infinite-state automaton**, in which bursts appear naturally as *state transitions*.

Another extension for this model is in the paper
> Zhao, Wayne Xin, et al. "Identifying event-related bursts via social media activities." Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning. Association for Computational Linguistics, 2012.

This paper extends the model in the first paper, making the model can handle multiple document streams. 

## Details

### Requirements

Python: 3.x

Packages: numpy(1.14.4), scipy(1.1.0).

### Demo

This is a demo using simple model in the KDD paper.

Number of state: 3 (See more details in impl/kdd_model.py) 

**input**

[10,70,60,60,70,10,10,119,120,13,10]

**output**

[0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

## Note

The project is being developed ...