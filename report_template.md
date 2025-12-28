# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : HADDAOU Hanna
- **Projet** : Projet 15, ESC-50 (50 classes) avec CNN 2D sur spectrogrammes log-mel
- **Dépôt Git** : https://github.com/hannahadd/csc8607_projects
- **Environnement** : python == 3.11, torch == ..., torchaudio == ..., cuda == False (local Mac / MPS)
- **Commandes utilisées** :
  - Entraînement : python -m src.train --config configs/config.yaml
  - LR finder : python -m src.lr_finder --config configs/config.yaml
  - Grid search : python -m src.grid_search --config configs/config.yaml
  - Évaluation : python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) :
- **Type d’entrée** (image / texte / audio / séries) : audio (fichiers WAV) transformé en spectrogramme log-mel (traité comme une image)
- **Tâche** (multiclasses, multi-label, régression) : classification multiclasses (50 classes)
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : (1, 64, T)
- **Nombre de classes** (`meta["num_classes"]`) : 50

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

J’utilise le dataset ESC-50, un jeu de données de classification de sons environnementaux contenant 2 000 enregistrements audio de 5 secondes répartis en 50 classes (40 exemples par classe). Les données sont fournies sous forme de fichiers WAV et d’un fichier d’annotations meta/esc50.csv (colonnes notamment : filename, fold, category). Chaque audio est converti en spectrogramme log-mel afin d’être traité par un CNN 2D.

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
|------:|----------:|--------------------------------------------------------|
| Train |     1200      |                           sons de 5s                             |
| Val   |       400    |                              mêm distribution attendues que train/test                          |
| Test  |        400   |                                   split indépendant via fold 5                     |

**D2.** Donnez la taille de chaque split et le nombre de classes.  
Avec cette stratégie par folds, on obtient 1200 exemples en train, 400 en validation et 400 en test, pour un total de 50 classes. Les entrées sont des spectrogrammes de forme (1, 64, T).

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).
Je n’ai pas créé de split aléatoire : j’ai utilisé le protocole du dataset via les folds du fichier esc50.csv. Cela garantit une séparation reproductible et standard. (La seed n’intervient donc pas dans la création des splits.)

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement. 

Le dataset ESC-50 est conçu pour être équilibré (40 exemples par classe sur l’ensemble). Sur le split test (fold 5), la classe majoritaire observée (“pig”) représente 8/400 = 2%, ce qui confirme l’absence de déséquilibre marqué.
L’accuracy est une métrique pertinente ici et il n’est pas nécessaire d’utiliser des pondérations de classes ; le modèle doit dépasser largement ~2% pour démontrer un apprentissage réel.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).
Les enregistrements audio font 5 secondes, mais une fois transformés en spectrogrammes, on obtient une représentation temporelle (axe T) déterminée par les paramètres de STFT/hop. Les labels sont single-label (une classe par exemple). Aucune donnée manquante n’est attendue : chaque ligne du CSV référence un fichier WAV existant.

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

Chargement WAV -> conversion mono

- Vision : resize = __, center-crop = __, normalize = (mean=__, std=__)…
- Audio : resample = 16000 Hz, mel-spectrogram (n_mels= 64, n_fft=400 , hop_length=160), Conversion en échelle log avec AmplitudeToDB
- NLP : tokenizer = __, vocab = __, max_length = __, padding/truncation = __…
- Séries : normalisation par feature (par bande mel) : moyenne/écart-type calculés sur l’axe temps

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  

Ces prétraitements sont nécessaires pour :
standardiser le format audio (mono + 16 kHz) ;
produire une représentation temps-fréquence exploitable par un CNN (log-mel spectrogram) ;
stabiliser l’entraînement (AmplitudeToDB + normalisation).

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?
Les prétraitements sont identiques pour train/val/test (aucune opération aléatoire). Cela garantit que la validation et le test mesurent la performance sur des données traitées de façon déterministe.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

Augmentations (via src/augmentation.py, appliquées uniquement au train) :
Time shift (décalage temporel par roll) : max_shift_pct = 0.05, p = 0.3
Frequency masking : max_width = 8, num_masks = 1, p = 0.5
Time masking : max_width = 20, num_masks = 1, p = 0.5

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
Ces augmentations sont adaptées aux spectrogrammes : elles simulent de petites variations temporelles et une légère “occlusion” temps/fréquence (type SpecAugment), ce qui améliore généralement la robustesse du modèle sans modifier la classe.

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.
- un léger décalage temporel ne change pas la catégorie du son (même événement sonore).
- les masquages temps/fréquence retirent une petite partie d’information mais ne changent pas l’identité globale du signal (on ne modifie pas l’étiquette).

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
![alt text](image.png)
Les exemples montrent des spectrogrammes log-mel cohérents : une énergie concentrée sur certaines bandes de fréquences selon la classe, et des variations temporelles visibles sur l’axe T. Cela confirme que le pipeline “WAV → log-mel” produit des entrées exploitables pour un CNN 2D.

**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.
Forme d’un batch train : (batch_size, 1, 64, T): Cela est cohérent avec meta["input_shape"] = (1, 64, T) (une observation), le batch ajoutant simplement la dimension batch_size.
---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `accuracy` → score = `0.02` classe majoritaire : pig, 8/400 dans le test
- **Prédiction aléatoire uniforme** — Métrique : `accuracy` → score = `0.02`random uniforme sur 50 classes
J'ai: 
(csc8607_esc50) hanna@MacBook-Air-de-Hanna csc8607_esc50 % python src/baselines.py

TEST size: 400
Num classes: 50
Majority class: pig count: 8
Majority baseline accuracy: 0.02
Random-uniform baseline accuracy: 0.02


_Commentez en 2 lignes ce que ces chiffres impliquent._
Le test est quasiment équilibré (la classe la plus fréquente ne représente que 2% des exemples), donc la baseline “majority class” est très faible. Une prédiction uniforme atteint ~2% aussi : il faut donc que le modèle dépasse largement ce niveau pour montrer un apprentissage réel.

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ 3.912
- **Observée sur un batch** : `3.949`
- **Vérification** : backward OK, gradients ≠ 0 Loss initiale attendue (multi-classe) ≈ log(num_classes) = log(50) ≈ 3.912
Observée sur un batch : 3.949
Vérification : backward OK, gradients ≠ 0 (grad_norm_sum_l2 = 2.266)

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.
La loss initiale observée (3.949) est cohérente avec la valeur attendue pour une classification uniforme sur 50 classes (log(50) ≈ 3.912).
Forme d’un batch d’entrée : (64, 1, 64, 501).
Forme de sortie du modèle : (64, 50).

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = 32` 
- **Hyperparamètres modèle utilisés** (les 2 à régler) : channels=[32,64,128], kernel_size=3
- **Optimisation** : LR = `0.001`, weight decay = `0.0001` (0 ou très faible recommandé)
- **Nombre d’époques** : `200`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

![alt text](image-3.png)

![alt text](image-4.png)

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.
J’ai entraîné le modèle en mode overfit sur un sous-ensemble de N = 32 exemples (train). Les hyperparamètres imposés du modèle sont channels = [32, 64, 128] et kernel_size = 3. J’ai utilisé LR = 1e-3, weight decay = 1e-4, pendant 200 époques.
La courbe train/loss descend progressivement vers ~0 et, en parallèle, train/acc atteint ~1.0 : le modèle arrive donc à mémoriser parfaitement ces 32 exemples. Cette capacité à obtenir une loss quasi nulle et une accuracy quasi parfaite sur un tout petit échantillon est exactement ce qui prouve l’overfit (et valide que le pipeline + modèle + backprop fonctionnent).
---

## 4) LR finder

- **Méthode** : balayage LR (log-scale) sur 200 itérations, en loggant (lr, loss)
- **Fenêtre stable retenue** : 2.4e-4 → 6.2e-2 (la loss baisse régulièrement. au-dessus, ça devient instable et remonte — ex. lr≈2.5e-1 puis 1.0)
- **Choix pour la suite** :
  - **LR** = `1e-2`
  - **Weight decay** = `1e-4` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

![alt text](image-5.png)

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.
La loss diminue nettement quand le LR passe dans la zone ~1e-3–1e-2 et atteint un minimum autour de 6e-2, puis elle remonte lorsque le LR devient trop grand (ex. 2.5e-1 et 1.0). Je choisis donc LR=1e-2, situé dans la zone de descente mais avec une marge de stabilité avant la zone instable. Le weight decay = 1e-4 apporte une régularisation légère pour limiter l’overfit sans freiner excessivement l’optimisation.
---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : {5e-3, 1e-2, 2e-2}
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A (channels) : {[32, 64, 128], [48, 96, 192]}
  - Hyperparamètre modèle B (kernel_size) : {3, 5}

- **Durée des runs** : `3` époques par run (1–5 selon dataset), même seed (42)

| Run (nom explicite) | LR    | WD     | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
|---------------------|-------|--------|-------|-------|-------------------------|----------|-------|
|                     |       |        |       |       |                         |          |       |
|                     |       |        |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

![alt text](image-7.png)
24 runs 

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).
Grâce qu fichier summary.csv on peut trouver la meilleure combinaison (selon best_val_acc) est :
Run : run18_k5_ch32-64-128_lr2p00e-02_wd1p00e-05
LR = 0.02
Weight decay = 1e-05
channels = [32, 64, 128]
kernel_size = 5
best_val_acc = 0.1725 (soit 17.25%)
epochs = 3, seed = 42

- channels contrôle la capacité du réseau : la configuration plus large ([48, 96, 192]) apprend généralement plus “agressivement” côté train (loss qui baisse plus vite), mais sur seulement 3 époques elle ne donne pas un gain de validation ici, ce qui suggère soit un besoin de plus d’époques pour en tirer profit, soit un début d’overfit/bruit plus élevé. La meilleure config reste donc la capacité plus modérée ([32, 64, 128]), plus stable pour un entraînement court.
- kernel_size influence le champ réceptif : avec k=5, les convolutions agrègent plus d’information temps/fréquence et donnent des courbes souvent plus stables (moins sensibles au bruit local). Ici, k=5 ressort comme meilleur que k=3 sur la validation, ce qui indique qu’un champ réceptif plus large aide la généralisation sur ESC-50 dans ce setup.

En résumé, sur ce mini grid search (3 époques), la meilleure perf validation est obtenue avec une capacité modérée et un kernel plus large, ce qui donne un compromis favorable entre vitesse d’apprentissage et stabilité sans sur-ajuster trop tôt.

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `0.02`
  - Weight decay = `1e-5`
  - Hyperparamètre modèle A = `channels = [32, 64, 128]`
  - Hyperparamètre modèle B = `kernel_size = 5`
  - Batch size = `64`
  - Époques = `15` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

![alt text](image-8.png)

![alt text](image-9.png)

Best val acc = 0.3425 (epoch 14)
Train acc fin = 0.4233
Commentaire (2–3 lignes) :
“train_loss baisse régulièrement, val_accuracy monte puis oscille, écart train/val modéré → apprentissage réel mais généralisation limitée.”

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


