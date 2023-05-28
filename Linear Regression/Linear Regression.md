# Linear Regression

Linear Regression is a **stadistical model** used to modelate the relationship between a dependent variable and one or more independent variables. It is based in the assumption that the relationship between the variables could be approximated by a **linear function**. The ecuation of a linear function is expressed as $y = mx + b$, where $y$ is the dependent variable, $x$ is the independent variable, $m$ is the slope of the line and $b$ is the interpolation of the line with the $y$ axis.

## Steps to build a Linear Regression model

1. **Data Collection**: Gather the data you want to use for the regression analysis. You need a dataset with two continuous variables: the independent variable (input or predictor) and the dependent variable (output or response).

2. **Data Preprocessing**: Ensure the data is clean and ready for analysis. Handle missing values, outliers, and perform any necessary data transformations.

3. **Data Splitting**: Split your dataset into two parts: the training set and the test set. The training set will be used to train the model, while the test set will be used to evaluate its performance.

4. **Model Training**: Use the training set to estimate the values. This process involves finding the best-fitting line that minimizes the error between the predicted and actual values.

5. **Model Evaluation**: Use the test set to evaluate the performance of your model. Common metrics for regression models include **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared** ($R^2$).

6. **Model Fine-tuning**: Depending on the evaluation results, you may need to fine-tune your model. This can involve adjusting hyperparameters, data transformations, or feature engineering.

7. **Prediction**: Once you are satisfied with the model's performance, you can use it to make predictions on new, unseen data.

## Interpretation of $y = mx + b$ in a neural network

- **y**: output variable.
- **m**: weight.
- **x**: input variable.
- **b**: bias.

## Limitations of Linear Regression

- Assumes a linear relationship between the variables, which may not be valid in all cases.
- May be sensitive to **outliers** and **nonlinear data**.
- Does not capture **non-linear relationships** or complex interactions between variables.
- Requires to meet the regression assumptions.

## Regression Assumptions

- **Independence of errors**:
This assumption states that the errors or residuals, which are the differences between the observed values and the values predicted by the regression model, must be independent of each other. In other words, there should be no systematic relationship or pattern in the residuals that suggests the existence of an unaccounted relationship in the model. If there is a dependence between the errors, it can bias the estimates and lead to incorrect conclusions.

- **Linearity of the relationship between variables**:
The linearity assumption states that the relationship between the dependent variable and the independent variables must be linear. This means that the functional form of the regression model should be appropriate to correctly represent how the dependent variable changes in response to changes in the independent variables. If the relationship is nonlinear and a linear model is fitted, biased and inaccurate estimates may be obtained.

- **Homoscedasticity of the errors**:
This assumption indicates that the spread of the errors should be constant across the entire range of the independent variables. In other words, the variability of the errors should be similar for all levels of the independent variables. If this assumption is violated and the errors have unequal variance, the model's predictions may be more precise in certain ranges of values and less precise in others, affecting the validity of inference
