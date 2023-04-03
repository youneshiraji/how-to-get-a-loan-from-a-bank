from sklearn.datasets import make_classification #Importation de la fonction permettant de générer des données pour les classifier
import pandas as pd #Importation de pandas pour la manipulation de données en tableau
import numpy as np #Importation de numpy pour les calculs numériques et pour les tableaux
import sklearn.tree as tree #Importation de la bibliothèque scikit-learn pour les arbres de décision
from sklearn import metrics #Importation de la bibliothèque scikit-learn pour les métriques de performance
from sklearn import preprocessing #Importation de la bibliothèque scikit-learn pour le prétraitement des données
from sklearn.metrics import classification_report, confusion_matrix #Importation de la fonction pour le rapport de classification ( Celui s'affichant avec les métriques)  et de la matrice de confusion pour évaluer la performance du modèle
from imblearn.under_sampling import TomekLinks #Importation de la fonction Tomek Links de la bibliothèque imblearn pour le sous-échantillonnage (undersampling)
from sklearn.model_selection import train_test_split, GridSearchCV #Importation des fonctions pour la séparation des données d'entraînement et de test et le GridSearch pour optimiser le choix des hyperparametres
from sklearn.linear_model import LogisticRegression #Importation du modèle de régression logistique
from sklearn.svm import SVC #Importation du modèle SVM
from sklearn.neighbors import KNeighborsClassifier #Importation du modèle de classification des KNN
from sklearn.tree import DecisionTreeClassifier #Importation du modèle d'arbre de décision
import matplotlib.pyplot as plt #Importation de la bibliothèque pour tracer les graphiques
import seaborn as sns #Importation de la bibliothèque pour les graphiques plus avancés
import matplotlib
matplotlib.use('TkAgg')
#Fixe une "RuntimeError" qui ne laisse pas le plt.show() fonctionner correctement


#Après certaines études, nous avions jugé qu'omettre la première ligne n'aurait pas de conséquence dramatique sur notre programme
#et demeurait une très bonne estimation .
donnees = pd.read_csv('german.csv')
adonnees=pd.read_csv('german.csv')
#Fonction qui permet la visualisation du fichier csv envoyé
Nombredesamples=donnees['1.1'].value_counts()
print("Le nombre respectif de 1 et de 2 \n"+str(Nombredesamples))
#Affiche le nombre de samples (Ce qui nous permet de voir la distribution de "1" et de "2" non équirépartis)
histogramme=donnees.hist(column='1.1', bins=5)
plt.title(f"Nombre de 1 et de 2 (Respectivement de bons et mauvais clients)")
plt.show()
donnees = donnees.to_numpy()

#Transformation du tableau de données (.csv) en un tableau array (permettant ainsi un meilleur contrôle)

X, y = np.asarray([]), []
#Initialisation des deux tableaux ( X :999x20, Y:1x20 )
for i in donnees:
    X = np.append(X, i[0:20])
for i in donnees:
    y.append(i[20])
#Remplir X des observations des colonnes de 1 à 19, et Y des observations de la colonne 20
X = np.array(X).reshape(-1, 20)
#Rendre X de dimensions 999x20
y = np.array( y )

enc = preprocessing.OrdinalEncoder()
enc.fit(X)
X = enc.transform(X)
#La fonction précédente s'occupe d'encoder les variables catégorielles (A114 ...) en leur attribuant des valeurs numériques.
def maximum(L):
    a=0
    for i in range(len(L)):
        if L[i][2] > a:
            a = L[i][2]
            b=i
    return L[b]
# Comme son nom l'indique, la fonction maximum s'occupe de trouver les meilleurs hyperparametre (Nous sommes conscients qu'il y a une fonction grid qui pourrait le faire a notre place, mais apparemment la recherche doit se faire "manuellement")
X = preprocessing.StandardScaler().fit(X).transform(X)
#La fonction précédente elle à son tour s'occupe de normaliser la matrice X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#Code s'occupe d'entrainer la machine, le 0.2 veut dire que le programme consacrera 20% des observations à l'entrainement et les autres 80% pour les test
tl = TomekLinks(sampling_strategy='auto', n_jobs=-1)
X_train, y_train = tl.fit_resample(X_train, y_train)
# L'undersampling utilisé afin d'équirépartir les observations (Le nombre de 1 ne désavantage pas le nombre de 2)

#Les différents modèles étudiés
models = [
{
        'name': 'Logistic Regression',
        'estimator': LogisticRegression(),
        'hyperparameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            #'C' est un paramètre de régularisation qui contrôle l'inverse de la force de régularisation, alors que 'solver' est un algorithme permettant d'optimiser la minimisation de la fonction de coût
        }
    },
{
        'name': 'SVM',
        'estimator': SVC(),
        'hyperparameters': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4],

        #La fonction noyau est utilisée pour transformer nos données en un espace supérieur contenant nos même données afin d'être plus apte à tracé l'hyperplan.
        #Le 'gamma' définit la portée de l'influence d'un seul exemple d'apprentissage. Plus la valeur est faible, plus la portée d'un point d'entraînement est grande.
        #'degree' est un argument de la fonction noyau polynomiale spécifiant le degré polynomial.

        }
    },

{
        'name': 'Decision Tree',
        'estimator': DecisionTreeClassifier(),
        'hyperparameters': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 1, 2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            # 'criterion' nous laisse le choix de choisir le critère sur lequel l'arbre se basera pour faire des feuilles.
            # 'max_depth' est le maximum de pronfondeur que pourra atteindre notre arbre.
            #"min_samples_split" Le nombre minimum d'échantillons requis pour diviser un nœud.
            #'min_samples_leaf' Il s'agit du nombre minimal d'échantillons nécessaires pour qu'une feuille puisse être formée.
        }
    },

{
        'name': 'KNN',
        'estimator': KNeighborsClassifier(),
        'hyperparameters': {
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2],
            # 'n_neighbors' le nombre de voisins à prendre en compte lors de la classification.
            # "weights" permet de définir une méthode pour pondérer les plus proches voisins.
            #'algorithm' est l'algorithme utilisée pour trouver les plus proches voisins.
            # 'p' nous donne le choix entre calculer les distances en norme 1 ou en norme 2.

        }
    },
]

L=[]
for model in models:
    print('-' * (len(model['name'])+4))
    print("| "+model['name']+" |")
    print('-' * (len(model['name'])+4))
    #Un peu d'esthétique dans le code ^^

    grid_search = GridSearchCV(
        estimator=model['estimator'],
        param_grid=model['hyperparameters'],
        scoring='accuracy',
        cv=5,
        verbose=0,
        n_jobs=-1
    )
    #La recherche des hyperparametres pour l'optimisation du modele

    grid_search.fit(X_train, y_train)
    resultats = grid_search.cv_results_
    for i, param in enumerate(resultats['params']):
        L.append(["Paramètres: "+ str(param),"Train accuracy", resultats['mean_test_score'][i]])

        #Affiche les resultats pour chaque hyperparametre (Aulieu de faire la recherche du hyperparametre automatique grace à la fonction best_parameter
        #Nous avions pensé à le faire manuellement comme demandé comme-ci)
    print(L)

    print("Voici le maximum d'Accuracy"+str(maximum(L)))
    L=[]
    prediction = grid_search.predict(X_test)
    Classification=classification_report(y_test, prediction)
    print(Classification)
    # Affiche l'ensemble des métriques de chaque modèle
    Ks = 10
    AccuracyMoyenne = np.zeros((Ks - 1))
    AccuracySTD = np.zeros((Ks - 1))
    if model['name']=="KNN":
        for n in range(1, Ks):
            # Entrainement et test du modèle pour différent KS
            neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
            predictionKNN = neigh.predict(X_test)
            AccuracyMoyenne[n - 1] = metrics.accuracy_score(y_test, predictionKNN)
            #STD = contrôle l'écart type
            AccuracySTD[n - 1] = np.std(predictionKNN == y_test) / np.sqrt(predictionKNN.shape[0])

        plt.plot(range(1, Ks), AccuracyMoyenne, 'g')
        plt.fill_between(range(1, Ks), AccuracyMoyenne - 1 * AccuracySTD, AccuracyMoyenne + 1 * AccuracySTD, alpha=0.10)
        plt.fill_between(range(1, Ks), AccuracyMoyenne - 3 * AccuracySTD, AccuracyMoyenne + 3 * AccuracySTD, alpha=0.10, color="green")
        plt.legend(('Accuracy ', '+/- 1xecarttype', '+/- 3xecarttype'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Nombre de voisins (K)')
        plt.tight_layout()
        plt.show()
        #Permet de déterminer le meilleur K à choisir et l'affiche
        print("Pour le modèle KNN : La meilleure accuracy moyenne est de", AccuracyMoyenne.max(), "avec k=", AccuracyMoyenne.argmax() + 1)
    if model['name'] == 'Decision Tree':
        Arbredeprofondeur4 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        Arbredeprofondeur4.fit(X_train, y_train)
        tree.plot_tree(Arbredeprofondeur4)
        plt.title(f"Arbre de profondeur 4")
        plt.show()
        #Permet de visualiser l'arbre de décision avec une profondeur de 4 tout en affichant l'entropie et le gain à chaque étape
    if model['name'] == 'Logistic Regression':
        
        # fit logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # plot data points and decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Logistic Regression')
        plt.show()

        

    cm = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Matrice de Confusion du modèle {model['name']} ")
    plt.show()
    # Affiche un tableau de 4 cases affichant les VP FP FN VN en s'obscurissant quand il y a peu de personnes