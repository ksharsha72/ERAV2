# Part 1
## Manual calculation of Backpropogation in a neural netowk of 1 hidden layers, 1 input layer and 1 output layer


∂E/∂W5 = (ao1-t1) * ao1*(1-ao1) * ah1
∂E/∂W6=(ao1-t1) * ao1*(1-ao1) * ah2
∂E/∂W7=(ao2-t2) * ao2*(1-ao2) * ah1
∂E/∂W8(ao2-t2) * ao2*(1-ao2) * ah2
![image](https://github.com/ksharsha72/ERAV2/assets/90446031/139c7d3f-feb1-472a-9add-7eb80b6c5c43)


∂E/∂W1 = ((ao1-t1) * ao1 * (1-ao1)*w5*ah1*(1-ah1)*i1) + ((ao2-t2) * ao2 * (1-ao2)*w7*ah1*(1-ah1)*i1) 
∂E/∂W2= ((ao1-t1) * ao1 * (1-ao1)*w5*ah1*(1-ah1)*i2) + ((ao2-t2) * ao2 * (1-ao2)*w7*ah1*(1-ah1)*i2) 
∂E/∂W3= ((ao1-t1) * ao1 * (1-ao1)*w6*ah2*(1-ah2)*i1) + ((ao2-t2) * ao2 * (1-ao2)*w8*ah2*(1-ah2)*i1) 
∂E/∂W4= ((ao1-t1) * ao1 * (1-ao1)*w6*ah2*(1-ah2)*i2) + ((ao2-t2) * ao2 * (1-ao2)*w8*ah2*(1-ah2)*i2) 
![image](https://github.com/ksharsha72/ERAV2/assets/90446031/7a04d9c2-ae28-4dde-8143-6fab01d2375f)



![full_image](https://github.com/ksharsha72/ERAV2/assets/90446031/125eae51-f70b-47e0-be3a-9c23aa66779b)
