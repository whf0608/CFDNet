# Paper Reproduction Code

## Overview

This repository contains the code used to reproduce the results presented in the paper "Cross-Scenario Damaged Building Extraction Network: Methodology, Application, and Efficiency Using Single-Temporal HRRS Imagery" by Haifeng Wang, Wei He, Zhuohong Li and Naoto Yokoya.

## Table of Contents

- [Paper Information](#paper-information)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Paper Information

- **Title**: Cross-Scenario Damaged Building Extraction Network: Methodology, Application, and Efficiency Using Single-Temporal HRRS Imagery
- **Authors**: Haifeng Wang, Wei He, Zhuohong Li and Naoto Yokoya
- **Abstract**: The extraction of damaged buildings is of significant importance in various fields such as disaster assessment and resource allocation. Although multi-temporal-based methods exhibit remarkable advantages in detecting damaged buildings, a single-temporal extraction remains crucial in actual emergency response situations. However, single-temporal damaged building extraction at high resolution remote sensing (HRRS) encounters the following challenges: {\color {red} (i) morphological divergence of damaged building among different  disasters, and (ii) critically limited annotated datasets for damaged building assessment.} To address these challenges, we propose a novel correlation feature decomposition network (CFDNet) for damaged building extraction from single-temporal images. The main idea of our CFDNet is to decompose complex damaged building features into multiple attribute-features on the basis of the proposed correlation feature decomposition mechanism. The attribute-features represent five characteristics that are used to describe damaged buildings. To mitigate the issue of limited training data, we further propose a coarse-to-fine training strategy, prioritizing the training of attribute-features based on their respective significance. In detail, at the coarse training stage, the CFDNet is trained to decompose the damaged building segmentation task into the extraction of multiple attribute-features. {\color {red}At the fine training stage, specific attribute-features, such as building feature and damage feature, are trained using fine-training datasets}. We have evaluated CFDNet on several datasets that cover different types of disasters and have demonstrated its superiority and robustness compared with state-of-the-art methods. Finally, we also apply the proposed model for the damaged building extraction in areas historically affected by major disasters, namely, the Turkey–Syria earthquakes on 6 February 2023, Cyclone Mocha in the Bay of Bengal on 23 May 2023, and Hurricane Ian in Florida, USA in September 2022. Results from practical applications also emphasize the significant advantages of our proposed CFDNet.

## Installing

1. Clone the repository
   ```sh
   git clone  https://github.com/whf0608/CFDNet.git

2. Install dependencies
  ```sh
  pip install -r requirements.txt # or npm install, yarn install, etc. depending on your package manager

