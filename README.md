<h1> MULTILAYER PERCEPTRON </h1>

<p align="center"> 
  <img src="https://img.shields.io/badge/42-Project-black?style=for-the-badge&logo=42">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python"> 
  <img src="https://img.shields.io/badge/AI-Neural%20Network-purple?style=for-the-badge"> 
  <img src="https://img.shields.io/badge/From-Scratch-success?style=for-the-badge"> 
</p>

<h2> Description </h2>

Multilayer Perceptron est un projet de Machine Learning consistant à implémenter un réseau de neurones artificiels from scratch.

<h4> Objectifs: </h4>

•  Comprendre les réseaux de neurones </br>
•  Implémenter un MLP sans framework </br>
•  Manipuler les mathématiques du ML </br>
•  Appliquer la backpropagation </br>
•  Travailler avec un dataset réel </br>

Le modèle permet de prédire si une tumeur est bénigne ou maligne

<h4> C’est quoi un MLP ? </h4>

Un Multilayer Perceptron (MLP) est un réseau de neurones composé de:
•  une couche d’entrée </br>
•  une ou plusieurs couches cachées </br>
•  une couche de sortie </br>

Chaque neurone applique :
•  une somme pondérée </br>
•  une fonction d’activation </br>

C’est un modèle de type feedforward

<h2> Fonctionnement </h2>

<h4> Forward propagation </h4>
•  Les données passent de couche en couche </br>
•  Chaque couche transforme les données </br>
•  Une prédiction est produite </br>

<h4> Backpropagation </h4>
•  Calcul de l’erreur </br>
•  Propagation du gradient </br>
•  Mise à jour des poids </br>

Basé sur la descente de gradient

<h4> Dataset </h4>
•  Données médicales (cancer du sein) </br>
•  Classification: </br>
  &nbsp;&nbsp;&nbsp;•  M → Malignant </br>
  &nbsp;&nbsp;&nbsp;•  B → Benign </br>
</br>
⚠️ Nécessite : </br>
•  preprocessing </br>
•  normalisation </br>

<h2> Features </h2>
•  Implémentation from scratch </br>
•  Réseau multi-couches </br>
•  Backpropagation </br>
•  Gradient descent </br>
•  Softmax / fonctions d’activation </br>
•  Split training / validation </br>
•  Calcul de loss </br>

<h2> Fonctionnement </h2>
•  Initialisation des poids aléatoires </br>
•  Forward pass → prédiction </br>
•  Calcul de la loss </br>
•  Backpropagation </br>
•  Mise à jour des poids </br>

<h2> Installation </h2>

    git clone https://github.com/epraduro/multilayer_perceptron.git
    cd multilayer_perceptron

<h2> Split Training / Validation dataset </h2>

    python3 split_data.py <path_to_data.csv>

<h2> Entraînement </h2>

    python3 train.py 
⚠️ Il est possible de modifier les parametres d'entrainement:

    python3 train.py --e 300 --l 30 16 8 4 1 

  Ici on modifie le nombre d'epochs (dans cet exemple: 300 passages sur le dataset d'entrainement) et le nombre de couche cachée (les étapes intermédiaires entre l'entrée et la sortie du réseau de neurones).

<h4> Exemple de sortie </h4>

    Epoch 10/100 - loss: 0.12 - val_loss: 0.15
    Epoch 20/100 - loss: 0.08 - val_loss: 0.11

<h2> Prédiction </h2>

    python3 predict.py <data_file.csv> <model_file.npy>

</br>

<p align="center"> <img src="result_train.png" width="550"/> </p> 

<h2> Résultats </h2>
Modèle capable de généraliser sur données inconnues </br>
</br>
<p align="center"> <img src="learning_curves_train_2026-05-01_15-31-47.png" width="450"/>
<img src="learning_curves_accuracy_2026-05-01_15-31-47.png" width="450"/>

<h2> Bonus </h2>

Implémentation de plusieurs modèles entrainés avec des nombres de couches cachées différentes:

<p align="center"> <img src="curves_bonus.png" width="350"/> </p>
