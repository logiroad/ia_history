Ce répo est dédié à l'historisque des différents réseaux de neurones qui ont été créés chez Logiroad.

L'idée est de garder un historique sur ce qui a été fait et de pouvoir les utiliser facilement par chacun.

Une arborescence a été mise en place afin de suivre un "certain standard" :

- classifier
    - nom_général_significatif
        - nom_du_réseau
            - infos.txt
            - labels.txt
            - fichiers de configuration
- detector
    - nom_général_significatif
        - nom_du_réseau
            - infos.txt
            - labels.txt
            - fichiers de configuration
- segmentation
    - nom_général_significatif
        - nom_du_réseau
            - infos.txt
            - labels.txt
            - fichiers de configuration


**Il est important pour chaque nouveau modèle :**

* De créer un fichier **infos.txt** contenant toutes les informations utiles pour l'utilisation du réseau.
Voici un exemple (detector/Véhicules/custom_vehicles_yolov4) :

`datasets : base logiroad L2R Annotation (12/2021)`

`modèle : Yolo V4`

`fichier de poids pré-entrainé utilisé : yolov4.conv.137 (sur https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)`

`fichier de poids : //cass/Bibliotheque/TRAIN/2010/model.4.yolo`

`mAP : `

`Vitesse Inférence :`

`GPU / Compute Capabilities :`

`Version Tensor RT :`

`... et toutes autres informations que vous jugez utiles`

* De créer un fichier **labels.txt**, contenant l'ensemble du nom de chaque classe.

* D'intégrer le(s) fichier(s) de configuration.





L'idéal est aussi de mettre à jour le confluence pour avoir un tableau contenant l'ensemble des réseaux :
[Confluence](https://logiroad.atlassian.net/wiki/x/kgDnAQ)



