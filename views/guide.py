"""Page : guide d'utilisation et lexique des statistiques."""
import pandas as pd
import streamlit as st

from views.common import ACTION_HELP, ACTION_LABEL, ACTIONS, PAGES

st.title("Guide et lexique")
st.markdown('<p class="lead">Ce que fait l\'outil, comment bien filmer, et ce que veut dire chaque chiffre.</p>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.markdown("#### :material/route: En 4 étapes")
        st.markdown(
            "1. **Importer** une vidéo (fichier, lien YouTube ou vidéo déjà sur l'ordinateur).\n"
            "2. **Choisir la portion** à analyser : un set, un rallye, ou le match entier.\n"
            "3. **Repérer le terrain** : les 4 coins sont proposés automatiquement, corrige-les si besoin.\n"
            "4. **Lancer** : l'analyse tourne en arrière-plan puis ouvre le tableau de bord.")
        st.page_link(PAGES["upload"], label="Nouvelle analyse", icon=":material/upload:")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/videocam: Bien filmer")
        st.markdown(
            "- **Plan fixe** sur trépied : pas de zoom ni de mouvement de caméra.\n"
            "- **En hauteur, derrière une ligne de fond** (idéal) ou sur le côté, au niveau du filet.\n"
            "- **Terrain entier visible**, avec ses 4 coins.\n"
            "- **1080p, 30 images/s ou plus**, bonne lumière : la balle est petite et rapide.")

st.markdown("### Ce que calcule l'analyse")
st.markdown(
    "1. **Joueurs** : chaque silhouette est détectée avec son squelette (poignets, chevilles…) puis suivie "
    "d'image en image. Les morceaux de suivi d'un même joueur sont recollés, arbitres et juges de ligne écartés.\n"
    "2. **Terrain** : les 4 coins donnent la correspondance entre l'image et le terrain réel (9 × 18 m). "
    "Chaque joueur est placé en mètres ; le côté du filet donne son équipe.\n"
    "3. **Balle** : un modèle spécialisé la cherche sur chaque image, puis les détections sont reliées en "
    "trajectoires. Les balles immobiles (ballons de réserve, ramasseurs) et les détections isolées sont "
    "écartées ; les petits trous sont comblés en suivant la courbe de la balle.\n"
    "4. **Touches** : la balle change brusquement de trajectoire près des mains d'un joueur. Au filet, "
    "si deux joueurs de camps opposés sont à portée, on retient celui qui respecte les règles du volley "
    "(3 touches maximum par camp, jamais deux de suite pour le même joueur).\n"
    "5. **Rôles** : l'ordre des touches dans chaque possession donne service, réception, passe, attaque…\n"
    "6. **Sauts** : montée nette des chevilles puis retour au sol au même endroit.\n"
    "7. **Déplacements** : positions au sol lissées, 5 fois par seconde, en excluant les sauts.")

st.markdown("### Lexique")
st.markdown("#### Actions")
st.table(pd.DataFrame({"Action": [ACTION_LABEL[a] for a in ACTIONS],
                       "Définition": [ACTION_HELP[a] for a in ACTIONS]}).set_index("Action"))
st.markdown("#### Indicateurs")
st.table(pd.DataFrame([
    ("Rallye", "Suite de touches sans pause de plus de 4 s. Une vidéo commencée en plein échange commence en "
               "plein rallye : le service n'est alors pas vu."),
    ("Possession", "Suite de touches d'une même équipe avant que la balle passe de l'autre côté."),
    ("Passage de filet", "Changement de camp de la balle au cours d'un rallye."),
    ("Construction en 3 touches", "Possession jouée réception (ou défense) → passe → attaque."),
    ("Attaque en suspension", "Attaque touchée pendant un saut détecté du joueur."),
    ("Saut (cm)", "Hauteur déduite du temps de vol : h = g·t²/8. Ordre de grandeur, à ±15 cm près."),
    ("Distance (m)", "Distance parcourue au sol pendant la présence du joueur à l'image."),
    ("Rythme (m/min)", "Distance divisée par le temps de présence : compare des joueurs vus plus ou moins longtemps."),
    ("Présence (s)", "Temps pendant lequel le joueur est suivi sur le terrain."),
    ("Balle détectée", "Part des images où la balle est trouvée. Sous 75 %, des touches manquent."),
    ("Contact sans joueur", "Changement de trajectoire sans joueur à portée : sol, filet, antenne, ou joueur caché."),
    ("Possession suspecte", "Plus de 3 touches du même camp : une touche au filet a sans doute été attribuée "
                            "au mauvais camp."),
], columns=["Terme", "Définition"]).set_index("Terme"))

st.markdown("### Limites à connaître")
st.markdown(
    "- Les joueurs sont numérotés par le suivi (#1, #2…), **pas par leur numéro de maillot**. "
    "Un joueur longtemps caché peut réapparaître sous un autre numéro.\n"
    "- Au filet, attaquant et contreur sont souvent superposés à l'image : l'attribution de ces touches est la "
    "plus fragile.\n"
    "- Les rôles sont déduits de l'ordre des touches : une touche manquée décale les rôles de la possession.\n"
    "- Le point (qui gagne le rallye) n'est pas encore détecté.")

with st.expander("Utilisation en ligne de commande"):
    st.code("python analyze.py match.mp4 --start 60 --end 180 --title \"Match du samedi\"\n"
            "python analyze.py match.mp4 --reuse            # recalcul sans GPU après modification des seuils\n"
            "python analyze.py match.mp4 --reuse --reclick  # recliquer les coins du terrain\n"
            "python -m streamlit run app.py                 # interface", language="bash")
    st.caption("Les seuils sont regroupés dans CFG en haut de analyze.py.")
