
!pip install tensorflow tensorflow-federated tensorflow-privacy scikit-learn pandas matplotlib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer


from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
    data = pd.read_csv(fn)


X = data.drop(columns=['target'])  
y = data['target']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


dp_optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=256,
    learning_rate=0.01
)

model = create_model()
model.compile(optimizer=dp_optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_test, y_test), verbose=2)


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nClassification Report (TinyML + DP):")
print(classification_report(y_test, y_pred))
acc_tinyml = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc_tinyml * 100))


confidence_labels = []
for prob in y_pred_prob:
    if prob < 0.5:
        confidence_labels.append("Normal")
    elif prob >= 0.5:
        confidence_labels.append("High Risk Detected")

df_confidence = pd.DataFrame({'Actual': y_test.values, 'Predicted Probability': y_pred_prob.flatten(), 'Prediction': confidence_labels})
print("\nSample Output with Confidence Level:")
print(df_confidence.head(10))


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
acc_dt = dt_model.score(X_test, y_test)
print("\nDecision Tree Accuracy: {:.2f}%".format(acc_dt * 100))


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
acc_rf = rf_model.score(X_test, y_test)
print("Random Forest Accuracy: {:.2f}%".format(acc_rf * 100))


def preprocess(x, y):
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(16)

clients_data = [(X_train[i::5], y_train[i::5]) for i in range(5)]
federated_train_data = [preprocess(x, y) for x, y in clients_data]

def model_fn():
    keras_model = create_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=tf.TensorSpec(shape=(None, X_train.shape[1]), dtype=tf.float32),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

for round_num in range(1, 6):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f"FL Round {round_num} Accuracy: {metrics['binary_accuracy']:.4f}")

acc_fl = float(metrics['binary_accuracy'])


models = ['TinyML + DP', 'Decision Tree', 'Random Forest', 'Federated Learning']
accuracies = [acc_tinyml * 100, acc_dt * 100, acc_rf * 100, acc_fl * 100]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(70, 100)
plt.grid(axis='y')
plt.show()
