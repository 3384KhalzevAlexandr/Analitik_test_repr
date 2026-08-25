import numpy as np
from sklearn.linear_model import LinearRegression

# Импорт исходных данных
from data_config import X_data, Y_data

# Обучение модели линейной регрессии.
model = LinearRegression()
model.fit(X_data, Y_data)

# Данные прогноза: [Тактовая частота, Утилизация, Температура].
target = np.array([[2000, 40, 54]])
prediction = model.predict(target)

print(f"Прогнозируемый UDP Throughput: {prediction[0]:.2f} Мбит/с")
print(f"Уравнение: Y = {model.coef_[0]:.2f}*Частота + {model.coef_[1]:.2f}*Утилизация + {model.coef_[2]:.2f}*Температура + {model.intercept_:.2f}")