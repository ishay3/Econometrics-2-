Validating Friedman's Permanent Income Hypothesis: An ARDL-ECM Analysis of Irish Household Consumption (1999-2019)
Executive Summary
This report empirically validates Friedman's (1957) Permanent Income Hypothesis (PIH) using quarterly data on Irish household consumption and disposable income from 1999Q1 to 2019Q4. Employing the Autoregressive Distributed Lag (ARDL) bounds testing approach and Error Correction Model (ECM) framework, we find [STRONG/MODERATE/LIMITED] evidence supporting the PIH. Our results indicate:

Long-run income elasticity: [X.XXX], consistent with proportional consumption response to permanent income
Cointegration evidence: [YES/NO] - confirming a stable long-run equilibrium relationship
Speed of adjustment: [X.XXX] per quarter, suggesting [RAPID/GRADUAL] consumption smoothing behavior


1. Introduction
1.1 Background and Motivation
The consumption function lies at the heart of macroeconomic theory and policy. Friedman's (1957) Permanent Income Hypothesis revolutionized our understanding by proposing that consumption decisions depend on permanent income rather than current income. According to PIH:
Ct=k⋅YtPC_t = k \cdot Y^P_tCt​=k⋅YtP​
where:

CtC_t
Ct​ = consumption at time t

YtPY^P_t
YtP​ = permanent (expected long-run) income

kk
k = constant marginal propensity to consume out of permanent income


Key Implications:

Households smooth consumption over their lifecycle
Temporary income shocks have minimal consumption effects
Only permanent income changes significantly affect consumption
Short-run marginal propensity to consume < long-run MPC

1.2 Research Objectives
This study aims to:

Test the validity of PIH for Irish households using modern econometric techniques
Estimate the long-run income elasticity of consumption
Analyze consumption adjustment dynamics to income shocks
Assess policy implications for fiscal stimulus effectiveness

1.3 Contribution
We contribute to the literature by:

Applying ARDL bounds testing methodology robust to I(0)/I(1) regressors
Analyzing recent Irish data including the financial crisis period (2008-2010)
Providing comprehensive diagnostic testing and economic interpretation
Linking empirical findings to policy-relevant consumption behavior


2. Literature Review
2.1 Theoretical Framework
Friedman's PIH (1957): Distinguishes between:

Permanent Income: Expected long-run average income
Transitory Income: Temporary deviations from permanent income

Households base consumption decisions on permanent income, leading to consumption smoothing. This contrasts with the Keynesian consumption function where consumption depends on current income.
Life-Cycle Hypothesis (Modigliani & Brumberg, 1954): Complementary theory suggesting households plan consumption over their entire lifetime, smoothing consumption across periods of varying income.
2.2 Empirical Evidence
Supporting Evidence:

Hall (1978): Random walk hypothesis of consumption
Campbell & Deaton (1989): Consumption smoothing in aggregate data
Attanasio & Weber (1995): Microeconomic evidence for consumption smoothing

Challenges to PIH:

Excess Sensitivity: Campbell & Mankiw (1989) found 50% of consumers follow "rule of thumb" behavior
Liquidity Constraints: Zeldes (1989) showed borrowing constraints limit smoothing
Precautionary Savings: Carroll (1997) emphasized uncertainty's role

2.3 Irish Context
Ireland's economic characteristics make it an interesting case:

High economic growth during "Celtic Tiger" era (1995-2007)
Severe contraction during financial crisis (2008-2010)
Strong recovery post-2013
Open economy with significant external trade exposure

Previous studies on Irish consumption:

[Include relevant Irish studies if available]
Limited recent ARDL-based analysis for Ireland


3. Data and Methodology
3.1 Data Description
Source: European Central Bank (ECB) Statistical Data Warehouse
Variables:

Final Consumption Expenditure of Households (CtC_t
Ct​)


Seasonally adjusted using Census X-13
Expressed in millions of domestic currency
Current prices with fixed parity conversion


Gross Disposable Income of Households (YtY_t
Yt​)


Seasonally adjusted using Census X-13
Expressed in millions of domestic currency
Current prices with fixed parity conversion



Period: 1999Q1 - 2019Q4 (84 observations)
Data Transformations:

Natural logarithms taken: ln⁡Ct\ln C_t
lnCt​ and ln⁡Yt\ln Y_t
lnYt​
Allows interpretation of coefficients as elasticities
Stabilizes variance and reduces heteroskedasticity

3.2 Econometric Methodology
3.2.1 Unit Root Testing
Before cointegration analysis, we test for unit roots using:
Augmented Dickey-Fuller (ADF) Test:

Δyt=α+βt+γyt−1+∑i=1pδiΔyt−i+εt\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_tΔyt​=α+βt+γyt−1​+i=1∑p​δi​Δyt−i​+εt​

H0H_0
H0​: Series contains unit root (non-stationary)

H1H_1
H1​: Series is stationary


KPSS Test:

H0H_0
H0​: Series is stationary

H1H_1
H1​: Series contains unit root


Using both tests provides robust stationarity assessment (confirmatory approach).
3.2.2 ARDL Bounds Testing Approach
Why ARDL?

Allows mixed I(0) and I(1) variables
Estimates short-run and long-run relationships simultaneously
Superior small sample properties
Avoids pre-testing biases of Johansen cointegration

ARDL(p,q) Specification:

ln⁡Ct=α0+∑i=1pβiln⁡Ct−i+∑j=0qγjln⁡Yt−j+εt\ln C_t = \alpha_0 + \sum_{i=1}^{p} \beta_i \ln C_{t-i} + \sum_{j=0}^{q} \gamma_j \ln Y_{t-j} + \varepsilon_tlnCt​=α0​+i=1∑p​βi​lnCt−i​+j=0∑q​γj​lnYt−j​+εt​
Model Selection: Information criteria (AIC, BIC) determine optimal lag structure
Bounds Test (Pesaran et al., 2001):
Tests for cointegration by examining:

F-statistic for joint significance of lagged levels
t-statistic on error correction term

Critical values depend on:

Number of variables (k)
Deterministic components (constant, trend)
Sample size

Decision Rule:

F > I(1) bound → Cointegration exists
F < I(0) bound → No cointegration
I(0) < F < I(1) → Inconclusive

3.2.3 Error Correction Model (ECM)
If cointegration exists, we estimate the ECM representation:
Δln⁡Ct=α+∑i=1p−1βiΔln⁡Ct−i+∑j=0q−1γjΔln⁡Yt−j+λECMt−1+εt\Delta \ln C_t = \alpha + \sum_{i=1}^{p-1} \beta_i \Delta \ln C_{t-i} + \sum_{j=0}^{q-1} \gamma_j \Delta \ln Y_{t-j} + \lambda ECM_{t-1} + \varepsilon_tΔlnCt​=α+i=1∑p−1​βi​ΔlnCt−i​+j=0∑q−1​γj​ΔlnYt−j​+λECMt−1​+εt​
where:

ECMt−1=ln⁡Ct−1−θ0−θ1ln⁡Yt−1ECM_{t-1} = \ln C_{t-1} - \theta_0 - \theta_1 \ln Y_{t-1}ECMt−1​=lnCt−1​−θ0​−θ1​lnYt−1​
Key Parameters:

λ\lambda
λ: Speed of adjustment (should be negative and significant)

θ1\theta_1
θ1​: Long-run income elasticity

γj\gamma_j
γj​: Short-run income elasticities


Long-Run Elasticity:

θ1=∑j=0qγj1−∑i=1pβi\theta_1 = \frac{\sum_{j=0}^{q} \gamma_j}{1 - \sum_{i=1}^{p} \beta_i}θ1​=1−∑i=1p​βi​∑j=0q​γj​​
Half-Life of Adjustment:

HL=ln⁡(0.5)ln⁡(1−λ)HL = \frac{\ln(0.5)}{\ln(1-\lambda)}HL=ln(1−λ)ln(0.5)​
Time for half of disequilibrium to be corrected.
3.3 Diagnostic Testing
We perform comprehensive diagnostics:

Normality: Jarque-Bera test for residual normality
Serial Correlation: Ljung-Box test for autocorrelation
Heteroskedasticity: Breusch-Pagan test for constant variance
Stability: Parameter stability across sample periods
Specification: Ramsey RESET test for functional form


4. Empirical Results
4.1 Preliminary Analysis
Descriptive Statistics [INSERT FROM YOUR OUTPUT]:
VariableMeanStd DevMinMaxConsumption (€M)[X][X][X][X]Income (€M)[X][X][X][X]Log Consumption[X][X][X][X]Log Income[X][X][X][X]
Key Observations:

Both series show upward trend over sample period
Visible impact of 2008-2010 financial crisis
Strong co-movement suggests potential long-run relationship

4.2 Unit Root Test Results
Table 1: Stationarity Tests
VariableADF Statisticp-valueKPSS Statisticp-valueConclusionln⁡Ct\ln C_t
lnCt​[X.XXX][X.XXX][X.XXX][X.XXX]I(1)ln⁡Yt\ln Y_t
lnYt​[X.XXX][X.XXX][X.XXX][X.XXX]I(1)Δln⁡Ct\Delta \ln C_t
ΔlnCt​[X.XXX][X.XXX]--I(0)Δln⁡Yt\Delta \ln Y_t
ΔlnYt​[X.XXX][X.XXX]--I(0)
Interpretation:

Both log consumption and log income are non-stationary in levels
ADF fails to reject unit root hypothesis (p > 0.05)
KPSS rejects stationarity hypothesis (p < 0.05)
First differences are stationary (integrated of order 1)
Conclusion: Both series are I(1), justifying cointegration analysis

4.3 ARDL Model Selection
Model Selection Results [INSERT YOUR RESULTS]:

Optimal lag structure: ARDL([p], [q]) selected by AIC
AIC value: [X.XXX]

Alternative specifications considered:
[Show top 5 models from output]
4.4 ARDL Estimation Results
Table 2: ARDL Model Estimates [INSERT YOUR LATEX TABLE HERE]
Short-Run Dynamics:

Contemporaneous income effect: γ0=[X.XXX]\gamma_0 = [X.XXX]
γ0​=[X.XXX] ([interpretation])

Lagged consumption effects capture habit persistence and adjustment dynamics
[Discuss significant coefficients]

Long-Run Relationship:

Long-run income elasticity: θ1=[X.XXX]\theta_1 = [X.XXX]
θ1​=[X.XXX]

Interpretation: 1% increase in permanent income → [X.XX]% increase in consumption
[CONSISTENT/INCONSISTENT] with PIH prediction of near-unity elasticity
[COMPARE TO LITERATURE]



Speed of Adjustment:

Error correction coefficient: λ=[X.XXX]\lambda = [X.XXX]
λ=[X.XXX]
Half-life: [X.XX] quarters = [X.XX] years
Interpretation: [Households adjust X% of disequilibrium per quarter]

4.5 Cointegration Test Results
Table 3: ARDL Bounds Test
Test StatisticValue10% I(0)10% I(1)5% I(0)5% I(1)Decisiont-statistic[X.XX]-2.57-3.21-2.86-3.53[Decision]F-statistic[X.XX]3.023.513.624.16[Decision]
Interpretation:
[Explain whether cointegration is found and what this means for PIH]
4.6 Diagnostic Tests
Table 4: Diagnostic Test Results
TestStatisticp-valueResultJarque-Bera (Normality)[X.XX][X.XXX][Pass/Fail]Ljung-Box Q(10) (Serial Correlation)[X.XX][X.XXX][Pass/Fail]Breusch-Pagan (Heteroskedasticity)[X.XX][X.XXX][Pass/Fail]Variance Ratio (Stability)[X.XX]-[Stable/Unstable]
Overall Assessment:
[Discuss whether model passes diagnostic tests and any concerns]

5. Discussion and Interpretation
5.1 Evidence for Permanent Income Hypothesis
Finding 1: Long-Run Income Elasticity

Estimated elasticity of [X.XXX] is [close to/far from] unity
[Strong/Moderate/Weak] support for proportional consumption response
Possible explanations for deviations:

[Measurement issues]
[Liquidity constraints]
[Precautionary savings]



Finding 2: Cointegration

[Evidence/No evidence] of long-run equilibrium relationship
Supports concept of permanent income as long-run attractor
Consistent with forward-looking consumption behavior

Finding 3: Adjustment Dynamics

Speed of adjustment ([X.XXX]) indicates [rapid/gradual] consumption smoothing
Half-life of [X.XX] quarters suggests [interpretation]
[Consistent/Inconsistent] with PIH predictions

5.2 Short-Run vs Long-Run Responses
Short-Run Behavior:

Immediate income elasticity: [X.XXX]
Suggests [significant/limited] response to transitory income shocks
Evidence of [excess sensitivity/appropriate smoothing]

Long-Run Behavior:

Long-run elasticity: [X.XXX]
Indicates consumption eventually adjusts to permanent income changes
[Supports/Challenges] PIH predictions

Comparison:

Ratio (SR/LR): [X.XX]
[Interpretation of the gap between short and long-run responses]

5.3 The Irish Context
Period-Specific Observations:

Celtic Tiger Era (1999-2007):

Rapid income growth
[Did consumption smoothing occur?]


Financial Crisis (2008-2010):

Sharp income contraction
[Evidence of consumption smoothing or excess sensitivity?]
Potential liquidity constraints during crisis


Recovery Period (2013-2019):

Gradual income recovery
[Pattern of consumption adjustment]



5.4 Comparison with Literature
Our findings [align/contrast] with previous studies:

Campbell & Mankiw (1989): [comparison]
Attanasio & Weber (1995): [comparison]
Recent Irish studies: [comparison if available]

Possible reasons for differences:

Time period analyzed
Methodological approach
Data frequency and measurement


6. Policy Implications
6.1 Fiscal Policy Effectiveness
Temporary Stimulus Measures:

Short-run elasticity of [X.XXX] suggests limited effectiveness
Households recognize temporary nature, maintain consumption plans
Example: One-time tax rebates may have low consumption multiplier

Permanent Policy Changes:

Long-run elasticity of [X.XXX] indicates significant consumption response
Permanent tax cuts/transfers more effective at stimulating consumption
Policy persistence matters for consumption impact

6.2 Automatic Stabilizers

Gradual adjustment (half-life: [X.XX] quarters) implies:

Built-in consumption smoothing mechanism
Automatic stabilizers may be less necessary
But also: Economy takes time to fully adjust to shocks



6.3 Social Safety Nets

Evidence of consumption smoothing suggests:

Households generally able to self-insure against transitory shocks
But: Possible liquidity constraints during severe downturns
Targeted support for constrained households may be warranted




7. Limitations and Robustness
7.1 Data Limitations

Aggregate Data:

Household-level heterogeneity not captured
Different income groups may behave differently
Aggregation may mask liquidity-constrained households


Measurement Issues:

Seasonal adjustment procedures
Currency conversion effects
Real vs. nominal income perceptions


Sample Period:

Includes financial crisis - structural break?
Limited pre-Euro data
Post-2019 COVID period not included



7.2 Methodological Considerations

ARDL Specification:

Lag length selection sensitivity
Linear specification assumption
Constant parameter assumption


Cointegration Testing:

Small sample properties of bounds test
Critical value tables based on asymptotic distributions



7.3 Robustness Checks
Potential extensions:

Split-sample analysis (pre/post crisis)
Alternative lag specifications
Inclusion of wealth/interest rate variables
Non-linear specifications


8. Conclusion
This study provides [strong/moderate/limited] empirical support for Friedman's Permanent Income Hypothesis using Irish household data from 1999-2019. Our main findings are:
Key Results:

Long-run income elasticity of [X.XXX] indicates consumption responds [proportionally/less than proportionally/more than proportionally] to permanent income changes
[Evidence/No evidence] of cointegration between consumption and income, supporting the existence of a stable long-run equilibrium relationship
Speed of adjustment of [X.XXX] suggests households take approximately [X.XX] quarters to correct half of any deviation from equilibrium

Theoretical Implications:

[The PIH framework provides a reasonable/inadequate description of Irish consumption behavior]
[Consumption smoothing is evident/limited in the data]
[Forward-looking behavior is supported/challenged by the evidence]

Policy Recommendations:

Permanent policy changes more effective than temporary measures
Consider adjustment dynamics when forecasting consumption responses
Target support toward liquidity-constrained households during crises

Future Research:

Microeconomic panel data analysis to identify heterogeneity
Incorporate wealth effects and interest rates
Analyze post-COVID consumption patterns
Cross-country comparison within Eurozone


References

Friedman, M. (1957). A Theory of the Consumption Function. Princeton University Press.
Modigliani, F., & Brumberg, R. (1954). "Utility Analysis and the Consumption Function: An Interpretation of Cross-Section Data." In K. Kurihara (Ed.), Post-Keynesian Economics.
Hall, R. E. (1978). "Stochastic Implications of the Life Cycle-Permanent Income Hypothesis: Theory and Evidence." Journal of Political Economy, 86(6), 971-987.
Campbell, J. Y., & Mankiw, N. G. (1989). "Consumption, Income, and Interest Rates: Reinterpreting the Time Series Evidence." NBER Macroeconomics Annual, 4, 185-216.
Campbell, J. Y., & Deaton, A. (1989). "Why Is Consumption So Smooth?" Review of Economic Studies, 56(3), 357-373.
Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds Testing Approaches to the Analysis of Level Relationships." Journal of Applied Econometrics, 16(3), 289-326.
Attanasio, O. P., & Weber, G. (1995). "Is Consumption Growth Consistent with Intertemporal Optimization? Evidence from the Consumer Expenditure Survey." Journal of Political Economy, 103(6), 1121-1157.
Zeldes, S. P. (1989). "Consumption and Liquidity Constraints: An Empirical Investigation." Journal of Political Economy, 97(2), 305-346.
Carroll, C. D. (1997). "Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis." Quarterly Journal of Economics, 112(1), 1-55.

[Add more relevant references]

Appendix
A. Mathematical Derivations
[Include detailed derivations of key formulas]
B. Additional Statistical Tests
[Include supplementary test results]
C. Code Availability
All analysis code is available upon request and has been implemented in Python using:

statsmodels for econometric estimation
pandas for data manipulation
matplotlib and seaborn for visualization

D. Figures
Figure 1: [Comprehensive diagnostic plots]
Figure 2: [Interpretation and impulse response plots]
Figure 3: [Additional visualizations]