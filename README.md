MDFA Tutorial — GitHub Project

The MDFA Tutorial is a collection of exercises and case studies designed to introduce users to — and provide hands-on experience with — the MDFA (Multivariate Direct Filter Approach).



\####

Author: Marc Wildi — https://marcwildi.com

\####

Repository: https://github.com/wiaidp/MDFA-tutorial

\####

Background (references \& links): https://wiaidp.github.io/MDFA-tutorial/about

\####

Project Structure:

The project directory is organized into four sub-folders:



\#	Folder

1	Common Functions

2	Literature

3	Output

4	Tutorials



\####



Getting started: Open the R project by clicking the project icon located in the main MDFA-tutorial folder. This will launch the project in RStudio. From there, load any tutorial file from the Tutorials sub-folder and run the code. Tutorials are arranged in order of increasing complexity.



\####

About the MDFA:

The MDFA is a prediction framework built around two core ideas i) recognizing the structural diversity of forecasting problems, and ii) aligning the chosen methodology with the forecaster's specific research priorities.



\####

Prediction tasks can take many forms:



\-One-step-ahead forecasting — predicting the immediately next observation



\-Multi-step-ahead forecasting — projecting values further into the future



\-General weighted combinations of future observations (involving possibly bi-infinite filters) — as encountered in signal extraction, trend estimation, and seasonal adjustment.



Because these objectives differ fundamentally in structure, no single fixed criterion can adequately serve them all.



\####

The ATS Trilemma:

Forecasting inherently involves three partly competing goals:



I. Accuracy — correctly predicting future levels



II. Timeliness — avoiding undue delays or premature signals



III. Smoothness — suppressing spurious noise and erratic fluctuations



Together, these form the ATS trilemma: any improvement along one dimension inevitably comes at the cost of at least one of the others. There is no free lunch.



\####

What Makes MDFA Distinctive:

The MDFA integrates all three dimensions within a unified optimization framework, tailoring the criterion to the specific structure of the prediction problem at hand while explicitly accounting for ATS trade-offs inherent in forecasting practice.



\####

Key properties of the approach include:





Generality — classical linear forecasting methods emerge as special cases, which can then be refined to reflect specific research priorities (customization)



Interpretability — optimization criteria are grounded in clear, fundamental principles, yielding closed-form solutions that are uniquely determined



Transparency — unlike black-box methods, MDFA provides a direct window into the forecasting mechanism



\####

These qualities make the MDFA especially well-suited for settings where opacity is either prohibited — such as compliance-driven or regulatory environments — or simply undesirable, such as when a deeper understanding of the underlying forecasting logic is required.

