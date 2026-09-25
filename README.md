# Volley Analytics

Analyse automatique de matchs de volley à partir d'une vidéo en caméra fixe :

suivi des joueurs, détection de la balle, comptage des touches, des sauts, des rallyes et des possessions,

avec un tableau de bord Streamlit.

## Stack

- Joueurs : YOLO26m-pose + BoT-SORT (Ultralytics), filtrés sur le terrain par polygone

- Balle : YOLOv8 entraîné sur une balle de volley

- Touches : cassure de trajectoire de la balle à proximité des mains d'un joueur

- Sauts : montée des chevilles par rapport au sol, hauteur estimée par le temps de vol

- Interface : Streamlit + Altair

## Installation (Windows, GPU AMD RDNA4)

```

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchvision

pip install -r requirements.txt

```

Sur GPU NVIDIA ou CPU, installer torch depuis pytorch.org à la place.

## Utilisation

```

python analyze.py <video.mp4> --ball <modele_balle.pt>

python -m streamlit run dashboard.py

```

Au premier lancement, cliquer les 4 coins du terrain (fond gauche, fond droit, avant droit, avant gauche).

Les seuils sont dans `CFG` en haut de `analyze.py` ; `--reuse` réanalyse sans repasser par le GPU.

## Limites

- Les joueurs sont identifiés par l'ID du tracker, pas encore par le numéro de maillot.

- La détection des touches est heuristique : rebonds avant service et contres peuvent fausser le compte.

- La hauteur de saut est un ordre de grandeur.

## Licence

AGPL-3.0, pour rester compatible avec Ultralytics.

