Ce répo est dédié à l'historisque des différents réseaux de neurones qui ont été créés chez Logiroad.

L'idée est de garder un historique sur ce qui a été fait et de pouvoir les utiliser facilement par chacun.

**Il est important pour chaque nouveau modèle :**

* De créer un fichier **infos.txt** contenant toutes les informations utiles pour l'utilisation du réseau.
Voici un exemple :

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

* ** ATTENTION : Ne pas poster le fichier de poids pour éviter de saturer trop rapidement le gitlab. Il faut mettre le lien ou le répertoire partagé dans lequel
on peut retrouver ce fichier, dans le fichier infos.txt**



L'idéal est aussi de mettre à jour le confluence pour avoir un tableau contenant l'ensemble des réseaux :
[Confluence](https://logiroad.atlassian.net/wiki/x/kgDnAQ)



