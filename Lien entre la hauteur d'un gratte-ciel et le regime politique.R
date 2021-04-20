my_data=read.csv(file.choose(), header=TRUE, sep=",", dec=".")
my_data
options(max.print=.Machine$integer.max)
#permet d'avoir l'ensemble des lignes du dataset
my_data
attach(my_data)
where=(Democracy==0)
#La courbe rouge correspond aux dictatures
plot(density(Skyscraper[where]), col="red", 
     main="Distribution des gratte-ciels en fonction de leur hauteur", 
     xlab="hauteur en mètres",
     lwd=2)
where=(Democracy==1)
#La courbe verte correspond aux démocraties
lines(density(Skyscraper[where]), col="darkgreen", lwd=2)
#Dans le régime des dictatures les gratte-ciels sont regroupés et varient pricipalement entre 0 et 200m
choisis=sample(1:164, 114)
model=glm(Democracy ~ Skyscraper, family=binomial, data=my_data[choisis,])
model$coefficients
b0=-0.43425876 
b1 =0.00153297 
#Plus le gratte-ciel est haut plus les chances d'avoir une démocraties sont faibles
#Variable binaire 0 ou 1 à trouver, nous sommes dans le cadre d'une regression logistique

e=2.71828
#450 Taille random d'un gratte-ciel, l'algorithme va nous dire si ce gratte-ciel appartient a une démocratie ou a une dicatuture 
e**(b0+b1*450)/(1+e**(b0+b1*450))
predictions=predict(model, newdata=my_data[-choisis,], type="response")
predictions=ifelse(predictions>0.5, 1, 0)
predictions
#Tableau croisé des prédictions vs la réalité pour ces pays, permet de voir les vrais positifs et les faux négatifs
table(predictions, reality=Democracy[-choisis])/50
#62% de prédictions correctes et 38% incorrects

# Comparaison avec la liberté de la presse. Voir s'il y a un lien entre la liberté de la presse et le regime en place 

choisis=sample(1:164, 114)
model=glm(Democracy ~ PressF, family=binomial, data=my_data[choisis,])
model$coefficients
predictions=predict(model, newdata=my_data[-choisis,], type="response")
predictions=ifelse(predictions>0.5, 1, 0)
table(predictions, reality=Democracy[-choisis])/50

#80% de predictions correctes vs 20% d'erreurs 
 
