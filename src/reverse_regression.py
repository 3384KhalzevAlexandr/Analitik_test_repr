import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Импорт исходных данных
from data_config import X_data, Y_data, throughputs

# Разделяем данные по устройствам
X_var1 = X_data[:5, :]  
X_var2 = X_data[5:, :]  
Y_var1 = Y_data[:5]
Y_var2 = Y_data[5:]

# Модели для утилизации устройства 1 и 2
model_cpu1 = LinearRegression()
model_cpu1.fit(Y_var1.reshape(-1, 1), X_var1[:, 1])

model_cpu2 = LinearRegression()
model_cpu2.fit(Y_var2.reshape(-1, 1), X_var2[:, 1])

# Модели для температуры устройства 1 и 2
model_temp1 = LinearRegression()
model_temp1.fit(Y_var1.reshape(-1, 1), X_var1[:, 2])

model_temp2 = LinearRegression()
model_temp2.fit(Y_var2.reshape(-1, 1), X_var2[:, 2])

# Функция интерполяции для частоты 1700 ГГц
def interpolate_for_freq(throughput):
    # Данные частоты
    freq4 = 1700
    freq1, freq2 = 1500, 2600
    # Получение предсказываемых значений для введенной частоты для устройств 1 и 2
    cpu1 = model_cpu1.predict([[throughput]])[0]
    temp1 = model_temp1.predict([[throughput]])[0]
    cpu2 = model_cpu2.predict([[throughput]])[0]
    temp2 = model_temp2.predict([[throughput]])[0]
    # Коэффициент интерполяции
    k = (freq4 - freq1) / (freq2 - freq1)
    # Утилизация и температура для новой введенной частоты
    # Рассчитывается на основе данных устройств 1 и 2, а также их схожести с четвертым
    new_cpu = cpu1 + k * (cpu2 - cpu1)
    new_temp = temp1 + k * (temp2 - temp1)
    
    return new_cpu, new_temp

# Коррекция для малых значений Throughput
def correct_predictions(throughput):
    cpu_600, temp_600 = interpolate_for_freq(600)
    
    if throughput <= 600:
        cpu_corr = (throughput / 600) * cpu_600
        temp_corr = 25 + (throughput / 600) * (temp_600 - 25)
    else:
        cpu_corr, temp_corr = interpolate_for_freq(throughput)
    
    cpu_corr = max(0, min(100, cpu_corr))
    temp_corr = max(20, min(100, temp_corr))
    
    return cpu_corr, temp_corr

# Формирование таблицы результатов
results = []
for tput in throughputs:
    new_fin_cpu, new_fin_temp = correct_predictions(tput)
    results.append((tput, new_fin_cpu, new_fin_temp))

# Вывод результатов
print(f"{'UDP Throughput':<15} {'Утилизация ЦП':<15} {'Температура ЦП':<15}")
print("-" * 45)
for tput, cpu, temp in results:
    print(f"{tput:<15} {cpu:<15.1f} {temp:<15.1f}")