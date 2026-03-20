MDFA Tutorial — GitHub Project



The MDFA (Multivariate Direct Filter Approach) provides a unified framework for solving general prediction problems while simultaneously accommodating specific research priorities and objectives.



The MDFA Tutorial is a collection of exercises and case studies designed to introduce users to — and provide hands-on experience with — the MDFA. 

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



\-One-step-ahead forecasting — predicting the immediately next observation (prioritizes short-term accuracy)



\-Multi-step-ahead forecasting — projecting values further into the future (fits and extrapolates short- to medium-term dynamics)



\-General weighted combinations of future observations (involving possibly bi-infinite filters) — as encountered in signal extraction, trend or cycle estimation, and seasonal adjustment (emphasizes medium- and long-term components).



Because these objectives differ in structure, no single fixed criterion can adequately serve them all.



\####

The ATS Trilemma:

Forecasting inherently involves three partly competing goals:



I. Accuracy — correctly predicting future levels



II. Timeliness — avoiding undue delays or premature signals



III. Smoothness — suppressing spurious noise and erratic fluctuations



Together, these form the ATS trilemma.



\####

Efficient frontier:



In an MDFA-optimized predictor, any improvement along one ATS-dimension inevitably comes at the cost of at least one of the others — there is no free lunch. This is a direct consequence of MDFA defining and residing on the efficient frontier of the ATS trilemma. The classical MSE predictor represents a single point on this frontier, whereas MDFA extends the solution space to the full two-dimensional hyperplane.



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



\####

Tutorial:



The MDFA tutorial addresses the above methodological considerations through a comprehensive set of exercises. Among others, it replicates classical ARIMA/VARIMA forecasting, Wiener-Kolmogorov signal extraction, and the Hodrick-Prescott, Christiano-Fitzgerald, Hamilton, and Beveridge-Nelson filters within the (M)DFA framework. Once replicated, any of these designs can be further customized to improve trade-offs along the ATS performance dimensions. The tutorial covers both univariate and multivariate settings, parametric and non-parametric extensions, as well as stationary and non-stationary data-generating processes. In addition, it assesses in- and out-of-sample performances and overfitting, and introduces a novel set of regularization techniques specifically tailored to the (M)DFA framework. The tutorial is designed to be an engaging, instructive, and enjoyable experience throughout.

