The MDFA tutorial project on github is a collection of exercises and case studies introducing to (and working with) the MDFA -Multivariate Direct Filter Approach. 

Author: Marc Wildi (https://marcwildi.com)

Repository:
https://github.com/wiaidp/MDFA-tutorial

Background (references, links):
https://wiaidp.github.io/MDFA-tutorial/about


Project:
The project folder contains four sub folders: 1. Common functions, 2. Literature, 3. output, 4. Tutorials.

Start the tutorial: click on the R project icon (in the main folder MDFA-tutorial) to open the project in R Studio. Load any of the tutorials contained in the Tutorials sub folder into R Studio. Run the code. Tutorials are numbered in increasing order of complexity.

The MDFA  is a prediction approach that emphasizes the structure of generic prediction problems (one step-ahead, multi-steap ahead, signal extraction) and the importance of aligning forecasting methods with the forecaster’s research priorities. 

Prediction tasks may take several forms. One may seek to forecast the next observation (one-step-ahead), predict values further into the future (multi-step-ahead), or estimate weighted combinations of future observations. 
The latter arises naturally in applications such as signal extraction, trend estimation, and seasonal adjustment. Because these objectives differ in structure, no single fixed criterion can adequately address them all.

Forecasting also involves several partly competing goals: accuracy in predicting future levels, timeliness in avoiding delays or excessive anticipation, and smoothness in suppressing spurious noise. 
These goals form what we term the ATS trilemma: improvements in any one of the three components—Accuracy, Timeliness, and Smoothness—inevitably come at the expense of at least one of the others.

The MDFA integrates these dimensions within a unified optimization framework. The criterion is tailored to the structure of the prediction problem while explicitly incorporating the ATS trade-offs that 
characterize forecasting practice. The approach is highly general: classical linear forecasting methods can be reproduced as special cases and subsequently refined to reflect specific research priorities. 
It is also interpretable in the sense that the optimization criteria follow simple, fundamental principles, yielding closed-form and uniquely determined solutions. 
These properties are particularly valuable in settings where black-box methods are either prohibited — as in applications subject to compliance or regulatory requirements — or simply undesirable, 
such as when the user seeks a deeper understanding of the forecasting mechanism.

