import numpy as np
import root_pandas as rpd 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
plt.rcParams['font.size'] = 18

def g1(x):
	return 0*x

def g2(x):
	return -3/4 * x

def g3(x):
	return -5/4 *x

def performance(prediction,label):
	#calculates precision, recall and accuracy of given prediction and label
	prediction = np.array(prediction)
	
	label = np.array(label)
	right = prediction[prediction==label]
	wrong = prediction[prediction!=label]
	#number of tp fp tn and fn
	tp = len(right[right==1])
	fp = len(wrong[wrong==1])
	tn = len(right[right==0])
	fn = len(wrong[wrong==0])
	
	
	if tp==0 and fp==0: 
		precision=1
	else:
		precision = 1- tp/(tp+fp)
	if tp==0 and fn==0:
		recall = 0
	else:
		recall = tp/(tp+fn)
	if tp+tn==0:
		accuracy=0
	else:			
		accuracy = (tp+tn)/(tp+tn+fn+fp)		
	return precision,recall,accuracy

def proj(x_real,y_real,m_g):
	#projects points on a linear function with slope m_g
	xproj=((x_real + y_real*m_g)/np.sqrt((1+m_g**2))) 
	return xproj

def predict(projected_x,n_cuts,cut_min,cut_max):
	#binary prediction by different cuts
	cuts = np.linspace(cut_min,cut_max,n_cuts)
	pred_cuts=[]
	for i in range(len(cuts)):
		pred = [] #muss der cut auch auf die gerade projiziert werden?
		for x in projected_x: 
			if x < cuts[i]:
				pred.append(0)
			else:
				pred.append(1)
		pred_cuts.append(pred)
	return pred_cuts,cuts #returns list of predictionlists for each cut

def labelling(exa,label):
	#generates labels for data
	if label==1:
		labels = np.ones(len(exa))
	elif label ==0:
		labels = np.zeros(len(exa))
	return labels	

def aufg2():
	#a)
	P0 = rpd.read_root('zwei_populationen.root',key='P_0_10000')
	#print(P0)
	P1 = rpd.read_root('zwei_populationen.root',key='P_1')
	#print(P1)
	figure=plt.figure(figsize=(16,9))
	plt.scatter(P0['x'],P0['y'],label='P0',color='red',s = 1,rasterized=True)
	plt.scatter(P1['x'],P1['y'],label='P1',color='navy',s = 1,rasterized=True)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')
	plt.xlim(-15,20)
	#plt.ylim(-30,20)
	plt.legend(loc='best')
	plt.savefig('scatter_P0_P1.png',dpi=200)
	#plt.show()
	#b)
	
	xx = np.linspace(-15,20)
	plt.plot(xx,g1(xx),'k',label='g1',rasterized=True)
	plt.plot(xx,g2(xx),'g',label='g2',rasterized=True)
	plt.plot(xx,g3(xx),'c',label='g3',rasterized=True)
	plt.legend(loc='best')
	plt.savefig('scatter_with_lines.png',dpi=200)
	#plt.show()

	P0g1_x = proj(P0['x'],P0['y'],0)
	P0g2_x = proj(P0['x'],P0['y'],-3/4)
	P0g3_x = proj(P0['x'],P0['y'],-4/5)

	P1g1_x = proj(P1['x'],P1['y'],0)
	P1g2_x = proj(P1['x'],P1['y'],-3/4)
	P1g3_x = proj(P1['x'],P1['y'],-5/4)
	#plot projected points into scatterplot
	plt.plot(P0g1_x,g1(P0g1_x),'r^',markersize=4,rasterized=True)
	plt.plot(P0g2_x,g2(P0g2_x),'r^',markersize=4,rasterized=True)
	plt.plot(P0g3_x,g3(P0g3_x),'r^',markersize=4,rasterized=True)
	plt.plot(P1g1_x,g1(P1g1_x),'bo',markersize=4,rasterized=True)
	plt.plot(P1g2_x,g2(P1g2_x),'bo',markersize=4,rasterized=True)
	plt.plot(P1g3_x,g3(P1g3_x),'bo',markersize=4,rasterized=True)
	plt.savefig('projection.png',dpi=200)
	#plt.show()
	plt.clf()

	#1d histograms of projected points
	bins=50
	fig=plt.figure(figsize=(16,20))
	ax1 = fig.add_subplot(311)
	plt.hist(P0g1_x,bins=bins,rasterized=True,histtype='stepfilled',color='red',label='Signal',alpha=0.5)
	plt.hist(P1g1_x,bins=bins,rasterized=True,histtype='stepfilled',color='navy',label='Background',alpha=0.5)
	
	plt.ylabel(r'Anzahl')
	plt.xlim(-15,15)
	plt.setp(ax1.get_xticklabels(), visible=False)

	ax2 = fig.add_subplot(312,sharey=ax1,sharex=ax1)
	plt.hist(P0g2_x,bins=bins,rasterized=True,histtype='stepfilled',color='red',label='Signal',alpha=0.5)
	plt.hist(P1g2_x,bins=bins,rasterized=True,histtype='stepfilled',color='navy',label='Background',alpha=0.5)
	plt.ylabel(r'Anzahl')
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_yticklabels(), visible=False)

	ax3 = fig.add_subplot(313,sharey=ax1,sharex=ax1)
	
	plt.hist(P0g3_x,bins=bins,rasterized=True,histtype='stepfilled',color='red',label='Signal',alpha=0.5)
	plt.hist(P1g3_x,bins=bins,rasterized=True,histtype='stepfilled',color='navy',label='Background',alpha=0.5)
	plt.xlabel(r'$x$')
	plt.ylabel(r'Anzahl')

	plt.tight_layout()
	plt.savefig('hists_proj.png',dpi=200)
	#plt.show()
	plt.clf()

	#exit(0)
	#label data
	labels_P0 = labelling(P0['x'],1)

	labels_P1 = labelling(P0['x'],0)

	labels = np.array([labels_P0,labels_P1]).flatten()


	# ncuts for all
	ncuts=50
	#prediction and performance for g1=0
	example1_x = np.array([P0g1_x,P1g1_x]).flatten() #merge sets


	pred_g1,cuts_g1 = predict(example1_x,n_cuts=ncuts,cut_min=-10,cut_max=10)
	
	prec_g1=[]
	rec_g1=[]
	acc_g1=[]

	for i in range(len(pred_g1)):

		prec1 , rec1 , acc1 = performance(pred_g1[i],labels)
		prec_g1.append(prec1)
		rec_g1.append(rec1)
		acc_g1.append(acc1)

	#print(prec_g1,rec_g1,acc_g1)	

	plt.clf()
	plt.plot(cuts_g1,prec_g1,label='precision',rasterized=True)
	plt.plot(cuts_g1,rec_g1,label='recall',rasterized=True)
	plt.plot(cuts_g1,acc_g1,label='accuracy',rasterized=True)
	plt.legend(loc='best')
	plt.savefig('performace_g1.png',dpi=100)
	#plt.show()


	#prediction and performance for g2
	example2_x = np.array([P0g2_x,P1g2_x]).flatten() #merge sets
		

	pred_g2,cuts_g2 = predict(example2_x,n_cuts=ncuts,cut_min=-10,cut_max=10)
	

	prec_g2=[]
	rec_g2=[]
	acc_g2=[]

	for i in range(len(pred_g2)):

		prec1 , rec1 , acc1 = performance(pred_g2[i],labels)
		prec_g2.append(prec1)
		rec_g2.append(rec1)
		acc_g2.append(acc1)
		
	#print(prec_g2,rec_g2,acc_g2)
	plt.clf()
	plt.plot(cuts_g2,prec_g2,label='precision',rasterized=True)
	plt.plot(cuts_g2,rec_g2,label='recall',rasterized=True)
	plt.plot(cuts_g2,acc_g2,label='accuracy',rasterized=True)
	plt.legend(loc='best')
	plt.savefig('performace_g2.png',dpi=100)
	#plt.show()

	#prediction and performance for g3
	example3_x = np.array([P0g3_x,P1g3_x]).flatten() #merge sets
		

	pred_g3,cuts_g3 = predict(example3_x,n_cuts=ncuts,cut_min=-10,cut_max=10)
	

	prec_g3=[]
	rec_g3=[]
	acc_g3=[]

	for i in range(len(pred_g3)):

		prec1 , rec1 , acc1 = performance(pred_g2[i],labels)
		prec_g3.append(prec1)
		rec_g3.append(rec1)
		acc_g3.append(acc1)
		
	#print(prec_g3,rec_g3,acc_g3)
	plt.clf()
	plt.plot(cuts_g3,prec_g3,label='precision',rasterized=True)
	plt.plot(cuts_g3,rec_g3,label='recall',rasterized=True)
	plt.plot(cuts_g3,acc_g3,label='accuracy',rasterized=True)
	plt.legend(loc='best')
	plt.savefig('performace_g3.png',dpi=100)
	#lt.show()




if __name__ == '__main__':
	aufg2()
