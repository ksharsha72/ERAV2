# Part 1
## Manual calculation of Backpropogation in a neural netowk of 1 hidden layers, 1 input layer and 1 output layer as shown in below image

									
										
																
									
									
![image](https://github.com/ksharsha72/ERAV2/assets/90446031/441f2ecf-1ed0-44dd-bb09-5c184bfd09d5)

									
1) First We calculate the gradients [W5, W6, W7, W8] w.r.t to total loss E_TOTAL=E1+E2
2) When calculating weights with W5 , the part E2 is not dependent on W5 so partial derivate with total loss becomes zero on E2 similarly
3) When calculating weights with W6 , the part E2 is not dependent on W6 so partial derivate with total loss becomes zero on E2
4) When calculating weights with W7 , the part E1 is not dependent on W7 so partial derivate with total loss becomes zero on E1 similarly
5) When calculating weights with W8 , the part E1 is not dependent on W8 so partial derivate with total loss becomes zero on E1 



		∂E/∂W5 = (ao1-t1) * ao1*(1-ao1) * ah1		
		∂E/∂W6=(ao1-t1) * ao1*(1-ao1) * ah2		
		∂E/∂W7=(ao2-t2) * ao2*(1-ao2) * ah1		
		∂E/∂W8(ao2-t2) * ao2*(1-ao2) * ah2		
![image](https://github.com/ksharsha72/ERAV2/assets/90446031/c870e9fb-1560-407d-9a9a-c12f3e8ee8f3)

6) later on we calculate the weight [W1, W2, W3, W4] w.r. to total loss E_TOTAL=E1+E2
7) Now the network has multiple ways to back propogate we will divide it in to small steps for W1, so to explain formulas of individual derivatives will be given
8) 
9) 





		∂E/∂W1 = ((ao1-t1) * ao1 * (1-ao1)*w5*ah1*(1-ah1)*i1) + ((ao2-t2) * ao2 * (1-ao2)*w7*ah1*(1-ah1)*i1) 					
		∂E/∂W2= ((ao1-t1) * ao1 * (1-ao1)*w5*ah1*(1-ah1)*i2) + ((ao2-t2) * ao2 * (1-ao2)*w7*ah1*(1-ah1)*i2) 					
		∂E/∂W3= ((ao1-t1) * ao1 * (1-ao1)*w6*ah2*(1-ah2)*i1) + ((ao2-t2) * ao2 * (1-ao2)*w8*ah2*(1-ah2)*i1) 					
		∂E/∂W4= ((ao1-t1) * ao1 * (1-ao1)*w6*ah2*(1-ah2)*i2) + ((ao2-t2) * ao2 * (1-ao2)*w8*ah2*(1-ah2)*i2) 					
![image](https://github.com/ksharsha72/ERAV2/assets/90446031/193c30fe-f74e-4877-8358-275a6b66cdaf)





