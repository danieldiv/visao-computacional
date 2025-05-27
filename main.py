# # X = gato
# # Y = cachorro

# Acurácia é a proporção de acertos em relação ao total de amostras
# Acurácia = (acertos / total) * 100
# Precisão é a taxa de verdadeiros positivos, mede a proporção de acertos entre as classes positivas
# Precisão = VP / (VP + FP)
# Recall é a taxa de verdadeiros positivos, mede a capacidade de identificar corretamente as classes positivas
# Recall = VP / (VP + FN)
# F1-Score é a média harmônica entre precisão e recall, mede o equilíbrio entre eles
# F1-Score = 2 * (Precisão * Recall) / (Precisão + Recall)

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Classificadores
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

X = []
y = []
for label, folder in enumerate(["classe_x", "classe_y"]):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))

        # Original
        X.append(img.flatten())
        y.append(label)

        # inverter as imagens nao apresentou resultados significativos

        # ruido
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img_noise = cv2.add(img, noise)
        X.append(img_noise.flatten())
        y.append(label)

        # brilho e contraste
        img_bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
        X.append(img_bright.flatten())
        y.append(label)
X = np.array(X)
y = np.array(y)

print(f"Total de amostras: {len(X)}")

seed = 42  # Apenas para garantir reprodutibilidade, None = aleatório

# Treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=None
)

classifiers = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=seed),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=seed),
    "SVM Linear": SVC(kernel="linear"),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for idx, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Matriz de confusão (linha 0)
    confMat = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confMat, display_labels=["X", "Y"])
    disp.plot(ax=axes[0, idx], cmap=plt.cm.Blues, colorbar=False)
    axes[0, idx].set_title(name)

    if idx != 0:
        axes[0, idx].set_ylabel("")

    # Métricas de avaliação (mantém como está)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nClassificador: {name}")
    print(f"Acurácia: {accuracy*100:.2f}%")
    print(f"Precisão: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")

    # Curva ROC (linha 1)
    try:
        if hasattr(classifier, "decision_function"):
            y_score = classifier.decision_function(X_test)
        elif hasattr(classifier, "predict_proba"):
            y_score = classifier.predict_proba(X_test)[:, 1]
        else:
            y_score = None
        if y_score is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            axes[1, idx].plot(
                fpr, tpr, color="darkorange", lw=2, label=f"Área = {roc_auc:.2f}"
            )
            axes[1, idx].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            axes[1, idx].set_xlim([0.0, 1.0])
            axes[1, idx].set_ylim([0.0, 1.05])
            axes[1, idx].set_title(f"Curva ROC - {name}")
            axes[1, idx].legend(loc="lower right")
            axes[1, idx].grid(True)
            axes[1, idx].set_xlabel("Taxa de Falsos Positivos")

            if idx == 0:
                axes[1, idx].set_ylabel("Taxa de Verdadeiros Positivos")

    except Exception as e:
        axes[1, idx].set_title(f"Curva ROC - {name}\n(erro)")
        print(f"Não foi possível plotar curva ROC para {name}: {e}")

plt.tight_layout()
plt.savefig("classifiers_performance.png")
plt.suptitle("Desempenho dos Classificadores", fontsize=16)
plt.subplots_adjust(top=0.9)  # Ajusta o título para não sobrepor os subplots
plt.show()
