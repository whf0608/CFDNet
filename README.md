# Paper Reproduction Code

## Overview

This repository contains the code used to reproduce the results presented in the paper "Cross-Scenario Damaged Building Extraction Network: Methodology, Application, and Efficiency Using Single-Temporal HRRS Imagery" by Haifeng Wang, Wei He, Zhuohong Li and Naoto Yokoya.

## Table of Contents

- [Paper Information](#paper-information)
- [Installation](#installation)
- [Usage](#usage)
- [Result](#result)
- [Acknowledgments](#acknowledgments)

## Paper Information

- **Title**: Cross-Scenario Damaged Building Extraction Network: Methodology, Application, and Efficiency Using Single-Temporal HRRS Imagery
- **Authors**: Haifeng Wang, Wei He, Zhuohong Li and Naoto Yokoya
- **Abstract**: The extraction of damaged buildings is of significant importance in various fields such as disaster assessment and resource allocation. Although multi-temporal-based methods exhibit remarkable advantages in detecting damaged buildings, a single-temporal extraction remains crucial in actual emergency response situations. However, single-temporal damaged building extraction at high resolution remote sensing (HRRS) encounters the following challenges: (i) morphological divergence of damaged building among different  disasters, and (ii) critically limited annotated datasets for damaged building assessment. To address these challenges, we propose a novel correlation feature decomposition network (CFDNet) for damaged building extraction from single-temporal images. The main idea of our CFDNet is to decompose complex damaged building features into multiple attribute-features on the basis of the proposed correlation feature decomposition mechanism. The attribute-features represent five characteristics that are used to describe damaged buildings. To mitigate the issue of limited training data, we further propose a coarse-to-fine training strategy, prioritizing the training of attribute-features based on their respective significance. In detail, at the coarse training stage, the CFDNet is trained to decompose the damaged building segmentation task into the extraction of multiple attribute-features. At the fine training stage, specific attribute-features, such as building feature and damage feature, are trained using fine-training datasets. We have evaluated CFDNet on several datasets that cover different types of disasters and have demonstrated its superiority and robustness compared with state-of-the-art methods. Finally, we also apply the proposed model for the damaged building extraction in areas historically affected by major disasters, namely, the Turkey–Syria earthquakes on 6 February 2023, Cyclone Mocha in the Bay of Bengal on 23 May 2023, and Hurricane Ian in Florida, USA in September 2022. Results from practical applications also emphasize the significant advantages of our proposed CFDNet.

## Installing
1. Clone the repository
   ```sh
   git clone  https://github.com/whf0608/CFDNet.git

2. Install dependencies
  ```sh
  pip install -r requirements.txt
  ```

## Usage


## Result
1. Bay of Bengal cyclone region: Bay of Bengal cyclone region: We compare Maxar's high-resolution remote sensing satellite data for the Bay of Bengal on February 17, 2023 and May 23, 2023. In the disaster area of the Bay of Bengal cyclone, which covers 0.926 km2, we detect 520 damaged buildings, with a combined damage area of 58,908.73 m2.
   <img src="http://47.121.214.152:8080/data/BayofBenga.png" alt="Bay of Bengal cyclone region" width="646" height="368">
2. Turkoglu Earthquake region: Turkoglu Earthquake region: We illustrate Maxar's high-resolution satellite remote sensing data on February 22, 2023 and World Imagery Wayback historical imagery data for the Turkoglu earthquake region. In the 7.246 km2 area, we detect 61 damaged buildings with a total damage area of 5,083.39 m2.
    <img src="http://47.121.214.152:8080/data/Turkoglu.png" alt="Turkoglu Earthquake region" width="646" height="368">
4. Islahiye Earthquake region: Islahiye Earthquake region: We analyze the damage to buildings in the Islahiye region, which covers an area of 4.143 km2. Compared with the results annotated by Microsoft, we determine that the distribution of damaged buildings is more uniform. We detect 251 damaged buildings, showing significant differences from Microsoft s annotations.
    <img src="http://47.121.214.152:8080/data/Islahiye.png" alt="Islahiye Earthquake region" width="646" height="368">
6. Florida Hurricane region: Florida Hurricane region: We analyze the damaged buildings in a storm-affected area of 1.899 km2 in Florida. In this area, we detect 535 buildings with no damage and 381 damaged buildings, with a total detected damaged area of 92,376.69 m2.
    <img src="http://47.121.214.152:8080/data/Florida.png" alt="Florida Hurricane region" width="646" height="368">
