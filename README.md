# ift-7030-daytrading
Projet dans le cadre du cours IFT-7030 de traitement de signaux.

- extractor: Contient les scripts d'extraction. Ne peut pas être utilisé car il nécessite la BD PostgreSQL, un compte InteractiveBroker avec subscription au market data.
- ppolstm: On peut rouler un entraînement à partir du fichier main.py. Ce dernier lit les données boursières (un petit échantillon de 3 tickers/3 journées) dans data/common_10s_202312192103.csv. Il se peut que vous deviez changer le chemin du fichier (csv_file = 'ppolstm/data/csv/common_10s_202312192103.csv'). Le modèle n'est pas encore profitable donc ne soyez pas surpris que le profit ne soit pas encore intéressant.
