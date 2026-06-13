#Gerekli paketler
install.packages("Ecdat")
install.packages("caret")
install.packages("RWeka")
install.packages("e1071")
install.packages("class")
install.packages("cluster")

library(Ecdat)
library(caret)
library(RWeka)
library(e1071)
library(class)
library(cluster)

#veri setini cagiralim
data("Computers")

View(Computers)
head(Computers)    # ilk 6 gozlem
str(Computers)     # degisken tipleri
summary(Computers) # ozet istatistikler

# eksik veri kontrolu
anyNA(Computers)   # FALSE ise eksik veri yoktur

# hedef nitelik premium degiskenidir
Computers$premium <- as.factor(Computers$premium)

table(Computers$premium)

str(Computers)

# veri setini %70 egitim %30 test olacak sekilde ayiralim
set.seed(1)

egitimindisleri <- createDataPartition(y = Computers$premium, p = 0.70,list = FALSE)

egitim <- Computers[egitimindisleri,]

test <- Computers[-egitimindisleri,]

dim(egitim)
dim(test)

#--------------------------KARAR AGACI---------------------------------

# karar agaci modeli kurulur

kararagacimodeli <- J48(premium ~ ., data = egitim)

print(kararagacimodeli)

summary(kararagacimodeli)

plot(kararagacimodeli)


# tahmin yapalim

tahmin_kararagaci <- predict(kararagacimodeli, test)

# tahmin ve gercek degerleri karsilastiralim

data.frame(Gercek = test$premium, Tahmin = tahmin_kararagaci)

table(tahmin_kararagaci, test$premium, dnn = c("Tahmin","Gercek"))

confusionMatrix(tahmin_kararagaci,test$premium)

#-----------------------------KNN--------------------------------------

# KNN algoritmasi sayisal veriler ile calistigi icin
# kategorik degiskenler veri setinden cikartildi

knnveri <- Computers[,c(
  "price",
  "speed",
  "hd",
  "ram",
  "screen",
  "ads",
  "trend"
)]

# min-maks normalizasyonu

minmaks <- preProcess(knnveri, method = c("range"))

normalizeveri <- predict(minmaks,knnveri)

# hedef degisken eklenir

normalizeveri$premium <- Computers$premium

summary(normalizeveri)

# egitim-test ayiralim

set.seed(1)

egitimindisleri <- createDataPartition(y = normalizeveri$premium, p = .70, list = FALSE)

egitim <- normalizeveri[egitimindisleri,]

test <- normalizeveri[-egitimindisleri,]

testnitelikleri <- test[,-8]

testhedefnitelik <- test[[8]]

egitimnitelikleri <- egitim[,-8]

egitimhedefnitelik <- egitim[[8]]

# KNN modeli

k_degeri <- 10

dogruluk <- NULL

for(i in 1:k_degeri)
{
  set.seed(1)
  
  tahminisiniflar <- knn(
    egitimnitelikleri,
    testnitelikleri,
    egitimhedefnitelik,
    k = i
  )
  
  dogruluk[i] <- mean(
    tahminisiniflar == testhedefnitelik
  )
  
  dogruluk[i] <- round(
    dogruluk[i],
    2
  )
}

for(i in 1:k_degeri)
{
  print(
    paste(
      "k=",
      i,
      "icin elde edilen dogruluk=",
      dogruluk[i]
    )
  )
}

tablom <- table(tahminisiniflar, testhedefnitelik, dnn = c("tahmini siniflar","gercek siniflar"))

tablom

confusionMatrix(tahminisiniflar,testhedefnitelik)

#----------------------------NAIVE BAYES-------------------------------

# naive bayes modeli kuralim

naivebayes_modeli <- naiveBayes(egitimnitelikleri, egitimhedefnitelik)

# tahmin yapalim

tahminisiniflar <- predict(naivebayes_modeli, testnitelikleri)

# gercek ve tahmini degerleri karsilastiralim

data.frame(Gercek = testhedefnitelik, Tahmin = tahminisiniflar)

table(tahminisiniflar, testhedefnitelik,dnn = c("tahmini siniflar","gercek siniflar"))

confusionMatrix(tahminisiniflar,testhedefnitelik)

#------------------------------K-MEANS---------------------------------

# kmeans algoritmasi sadece sayisal veriler ile calisir

kmeansveri <- Computers[,c(
  "price",
  "speed",
  "hd",
  "ram",
  "screen",
  "ads",
  "trend"
)]

summary(kmeansveri)

# normalize ederek devam edelim

install.packages("clusterSim")
library(clusterSim)

normalizeveri <- data.Normalization(x = kmeansveri, type = "n4", normalization = "column")

# 2 kume icin model

set.seed(1)

model2 <- kmeans(x = normalizeveri, centers = 2)

table(model2$cluster)

model2$totss
model2$tot.withinss
model2$withinss
model2$betweenss

# 3 kume icin model

set.seed(1)

model3 <- kmeans(x = normalizeveri,centers = 3)

table(model3$cluster)

model3$totss
model3$tot.withinss
model3$withinss
model3$betweenss

# silhouette hesaplayalim

library(cluster)

k <- 10

silhouette_degeri <- 0

for(i in 2:k)
{
  set.seed(1)
  
  bilgisayar_modeli <- kmeans(
    normalizeveri,
    centers = i
  )
  
  bilgisayar_silhouette <- silhouette(
    bilgisayar_modeli$cluster,
    dist(normalizeveri, method = "euclidean")
  )
  
  silhouette_degeri[i] <- mean(
    bilgisayar_silhouette[,c("sil_width")]
  )
}

silhouette_degeri

plot(
  2:k,
  silhouette_degeri[2:k],
  col = "black",
  pch = 20,
  cex = 1,
  lty = "solid",
  xlab = "kume_sayisi(k)",
  ylab = "silhouette"
)

lines(
  2:k,
  silhouette_degeri[2:k]
)

# k=3 icin gorsellestirme

install.packages("fpc")
library(fpc)

plotcluster(normalizeveri,model3$cluster)

clusplot(normalizeveri,model3$cluster,color = TRUE,shade = TRUE,labels = 2)

