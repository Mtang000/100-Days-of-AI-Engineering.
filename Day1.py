from sklearn.linear_model import LinearRegression


X = [[1], [2], [3], [4], [5]]


y = [20, 40, 60, 80, 100]


model = LinearRegression()

model.fit(X, y)
print("Model has successfully learned the data!")

new_student_hours = [[2.5]]
prediction = model.predict(new_student_hours)

print(
    f"If a student studies for {new_student_hours} hours, the AI predicts a score of: {prediction[0]}")
