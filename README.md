# ift-7030-daytrading
Projet dans le cadre du cours IFT-7030 de traitement de signaux.

- extractor: Contient les scripts d'extraction. Ne peut pas être utilisé car il nécessite la BD PostgreSQL, un compte InteractiveBroker avec subscription au market data.
- ppolstm: On peut rouler un entraînement à partir du fichier main.py. Ce dernier lit les données boursières (un petit échantillon de 3 tickers/3 journées) dans data/common_10s_202312192103.csv. Il se peut que vous deviez changer le chemin du fichier (csv_file = 'ppolstm/data/csv/common_10s_202312192103.csv'). Le modèle n'est pas encore profitable donc ne soyez pas surpris que le profit ne soit pas encore intéressant.
- informer-bounds: Contient le notebook pour l'entraînement pour le modèle de forecasting (`forecasting.ipynb`) et des deux types de classification (`classification.ipynb` et `classification_v2.ipynb`).



- modele-indicateur: l'entrainement se fait en exécutant la commande `./ift7030-project.py <filename> cuda [<optionnal-user-tag>]`.   Présentement, l'entrainement se fait sur le fichier fournit en ligne de commande et les résultats sont générés en utilisant les données provenant du fichier `../common_10s_20231112213000-upst.csv` .     Le but étant d'éviter l'overfitting excessif.    Le nom du fichier est hard-codé dans le script.   Le nom de la variable est "reference_file".    Par manque de temps, ce nom de fichier n'a pas pu être rendu disponible en ligne de commande. 

exemple de commande `./ift7030-project.py common_10s_20231112213000-etsy cuda etsy`   . 



