from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = [
    [1, 8], [2, 7], [3, 9], [4, 5], [5, 8],
    [6, 6], [7, 8], [8, 7], [9, 6], [10, 8]
]

y = [15, 32, 55, 62, 78, 81, 93, 95, 97, 100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Calculating the error margin
error = mean_squared_error(y_test, predictions)

print("--- AI Evaluation Metrics ---")
print(f"Actual Test Scores:    {y_test}")
print(f"AI's Exam Predictions: {[round(p, 2) for p in predictions]}")
print(f"Average Error Margin (MSE): {round(error, 2)}")
print("-----------------------------")


custom_student = [[6.5, 8]]
custom_pred = model.predict(custom_student)
print(
    f"Predicted score for {custom_student[0][0]} hours of study and {custom_student[0][1]} hours of sleep: {round(custom_pred[0], 2)}")
