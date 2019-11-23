#labels
import pandas as pd
import numpy as np
'''
labelsmucontext = (pd.read_csv('./mudi_labels.csv', header=None).values[:,1]).astype(str)
labelsmuindivid = (pd.read_csv('./mudi_labels.csv', header=None).values[:,3]).astype(str)
labelsmuiage = (pd.read_csv('./mudi_labels.csv', header=None).values[:,5]).astype(str)
mudiLLDs = pd.read_csv('./mudi_context_LLDs.csv', header=None).as_matrix()
print(mudiLLDs.shape)

labelsmuiage = pd.factorize((pd.read_csv('./mudi_labels.csv', header=None).values[:,5]).astype(str))
mudiLLDs = np.delete(mudiLLDs,[502],axis=1)
mudiLLDs = np.column_stack((mudiLLDs,labelsmuiage[0]))
print(mudiLLDs.shape)
np.savetxt('./mudi_age_LLDs.csv',mudiLLDs,delimiter=",", fmt="%s")

labelsmuisex = pd.factorize((pd.read_csv('./labels_mudidogs_all.csv', ).values[:,5]).astype(str))

mudiLLDs = np.delete(mudiLLDs,[8020],axis=1)
mudiLLDs = np.column_stack((mudiLLDs,labelsmuisex[0]))
print(mudiLLDs.shape)
np.savetxt('./mudi_sex_raw.csv',mudiLLDs,delimiter=",", fmt="%s")

#mescalina 2015
labelsmecontext_str = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,3]).astype(str)
labelsmebreed_str = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,7]).astype(str)
labelsmesex_str = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,5]).astype(str)
labelsmeage_str = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,6]).astype(str)
labelsmeindvid_str = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,1]).astype(str)

meLLDs = pd.read_csv('./mescalina2015_500LLDs.csv').as_matrix()
print(meLLDs.shape)

labelsmeindivid = (pd.factorize(pd.read_csv('./mescalina_labels.csv',header=None).values[:,1]))[0]
meLLDs = meLLDs[:,:500].astype(str)
meLLDs = np.column_stack((meLLDs,labelsmeindivid))
print(meLLDs.shape)
np.savetxt('./mescalina_individ_LLDs.csv',meLLDs,delimiter=",", fmt="%s")

labelsmecontext = (pd.factorize(pd.read_csv('./mescalina_labels.csv',header=None).values[:,3]))[0]
meLLDs = meLLDs[:,:500].astype(str)
meLLDs = np.column_stack((meLLDs,labelsmecontext))
print(meLLDs.shape)
np.savetxt('./mescalina_context_LLDs.csv',meLLDs,delimiter=",", fmt="%s")

labelsmeage = (pd.factorize(pd.read_csv('./mescalina_labels.csv',header=None ).values[:,6]))[0]
meLLDs = meLLDs[:,:500].astype(str)
meLLDs = np.column_stack((meLLDs,labelsmeage))
print(meLLDs.shape)
np.savetxt('./mescalina_age_LLDs.csv',meLLDs,delimiter=",", fmt="%s")

labelsmesex = (pd.factorize(pd.read_csv('./mescalina_labels.csv',header=None ).values[:,5]))[0]
meLLDs = np.delete(meLLDs,[500],axis=1)
meLLDs = np.column_stack((meLLDs,labelsmesex))
print(meLLDs.shape)
np.savetxt('./mescalina_sex_LLDs.csv',meLLDs,delimiter=",", fmt="%s")

labelsmebreed = (pd.factorize(pd.read_csv('./mescalina_labels.csv',header=None ).values[:,7]))[0]
meLLDs = np.delete(meLLDs,[500],axis=1)
meLLDs = np.column_stack((meLLDs,labelsmebreed))
print(meLLDs.shape)
np.savetxt('./mescalina_breed_LLDs.csv',meLLDs,delimiter=",", fmt="%s")

#mescalina2017
me17LLDs = pd.read_csv('./mescalina2017_500LLDs.csv').as_matrix().astype(str)[:,:503]
me17LLDs = me17LLDs[:,:500].astype(str)
print(me17LLDs.shape)

labels17individ_str = pd.read_csv('./mescalina2017.csv', header=None).values[:,2].astype(str)
labels17context_str = (pd.read_csv('./mescalina2017.csv', header=None).values[:,1]).astype(str)
labels17breed_str = (pd.read_csv('./mescalina2017.csv', header=None).values[:,4]).astype(str)
labels17sex_str = (pd.read_csv('./mescalina2017.csv', header=None).values[:,3]).astype(str)


labelsme17individ = pd.factorize(pd.read_csv('./mescalina2017.csv', header=None).values[:,2],na_sentinel=-1)[0]
me17LLDs = np.column_stack((me17LLDs,labelsme17individ))
print(me17LLDs.shape)
np.savetxt('./mescalina2017_individ_LLDs.csv',me17LLDs,delimiter=",", fmt="%s")
me17LLDs = me17LLDs[:,:500].astype(str)

labelsme17context = pd.factorize((pd.read_csv('./mescalina2017.csv', header=None).values[:,1]).astype(str))[0]
me17LLDs = np.column_stack((me17LLDs,labelsme17context))
print(me17LLDs.shape)
np.savetxt('./mescalina2017_context_LLDs.csv',me17LLDs,delimiter=",", fmt="%s")
me17LLDs = me17LLDs[:,:500].astype(str)

labelsme17breed = pd.factorize((pd.read_csv('./mescalina2017.csv', header=None).values[:,4]).astype(str))[0]
me17LLDs = np.column_stack((me17LLDs,labelsme17breed))
print(me17LLDs.shape)
np.savetxt('./mescalina2017_breed_LLDs.csv',me17LLDs,delimiter=",", fmt="%s")
me17LLDs = me17LLDs[:,:500].astype(str)

labelsme17sex = pd.factorize((pd.read_csv('./mescalina2017.csv', header=None ).values[:,3]).astype(str))[0]
me17LLDs = np.column_stack((me17LLDs,labelsme17sex))
print(me17LLDs.shape)
np.savetxt('./mescalina2017_sex_LLDs.csv',me17LLDs,delimiter=",", fmt="%s")
me17LLDs = me17LLDs[:,:500].astype(str)


#labelsurb = (pd.read_csv('./urban_labels.csv', header=None).values[:,1]).astype(str)
'''

mudiLLDs = pd.read_csv('./mudi_context_LLDs.csv', header=None).as_matrix()[:,0:500].astype(str)
meLLDs = pd.read_csv('./mescalina_context_LLDs.csv', header=None).as_matrix()[:,0:500].astype(str)
me17LLDs = pd.read_csv('./mescalina2017_context_LLDs.csv', header=None).as_matrix()[:,0:500].astype(str)
'''
labelsmucontext = (pd.read_csv('./mudi_labels.csv', header=None).values[:,1]).astype(str)
labelsmucontext = np.reshape(labelsmucontext,(len(labelsmucontext),1))
labelsmecontext = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,3]).astype(str)
labelsmecontext = np.reshape(labelsmecontext,(len(labelsmecontext),1))
labels17context = (pd.read_csv('./mescalina2017_original.csv', header=None).values[:,1]).astype(str)
labels17context = np.reshape(labels17context,(len(labels17context),1))
all_context_labels_str =np.vstack((labelsmucontext,labelsmecontext,labels17context))
print(np.column_stack(np.unique(all_context_labels_str,return_counts=True)))
all_context_LLDs=np.vstack((mudiLLDs,meLLDs,me17LLDs))

all_context_labels = pd.factorize(np.reshape(all_context_labels_str,(len(all_context_labels_str),)))[0]
all_4context_labels = []
for word in all_context_labels_str:
    if word in ['L-S1','L-S2','L-S3','stranger','fight'] :
        word = 'NA'
    if word in ['L-A','L-TA','alone',]:
        word = 'ND'
    if word in ['L-H','L-P','ball','food','play']:
        word = 'PA'
    if word in ['walk','L-PA']:
        word = 'PD'
    if word in ['L-D','L-O']:
        word = '?'
    all_4context_labels.append(word)
    
#del(labelsmucontext,labelsmecontext,labels17context)   
#del(mudiLLDs,meLLDs,me17LLDs)

all_4context_labels_str = np.reshape(all_4context_labels,(len(all_4context_labels),1))
print(np.column_stack(np.unique(all_4context_labels_str,return_counts=True)))
all_4context_LLDs = np.column_stack((all_context_LLDs,all_4context_labels_str))
all_4context_LLDs = all_4context_LLDs[~(all_4context_LLDs[:,500]=='?'),:]
all_4context_labels = np.reshape(pd.factorize(all_4context_LLDs[:,500])[0],(19136,1))
all_4context_LLDs = np.column_stack((all_4context_LLDs[:,:500],all_4context_labels))
np.savetxt('./all_4context_LLDs.csv',all_4context_LLDs,delimiter=",", fmt="%s")

all_context_LLDs = np.column_stack((all_context_LLDs,all_context_labels))
np.savetxt('./all_context_LLDs.csv',all_context_LLDs,delimiter=",", fmt="%s")

all_contexteven_LLDs = []
all_context_LLDs = np.column_stack((all_context_LLDs[:,:500],all_context_labels_str))
np.random.shuffle(all_context_LLDs)
for row in range(len(all_context_LLDs[:,])):
    overcount = False
    all_contexteven_LLDs.append(all_context_LLDs[row,:])    
    a ,b = np.unique(np.asarray(all_contexteven_LLDs)[:,500],return_counts=True) 
    print(a)
    print(b)
    for count in b:
        if count > 200:
            overcount = True
    if overcount == True:
        del[all_contexteven_LLDs[len(all_contexteven_LLDs)-1]]

all_contexteven_LLDs = np.asarray(all_contexteven_LLDs)
np.savetxt('./all_contexteven_LLDs.csv',all_contexteven_LLDs,delimiter=",", fmt="%s")

#check



all_context_LLDs = np.column_stack((all_context_LLDs,all_context_labels))
np.savetxt('./all_context3_LLDs.csv',all_context_LLDs,delimiter=",", fmt="%s")

'''
labelsmuindivid = (pd.read_csv('./mudi_labels.csv', header=None).values[:,3]).astype(str)
labelsmuindivid = np.reshape(labelsmuindivid,(len(labelsmuindivid),1))
labelsmeindivid = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,1]).astype(str)
labelsmeindivid = np.reshape(labelsmeindivid,(len(labelsmeindivid),1))
labels17individ = (pd.read_csv('./mescalina2017.csv', header=None).values[:,2]).astype(str)
labels17individ = np.reshape(labels17individ,(len(labels17individ),1))
all_individ_labels=np.vstack((labelsmuindivid,labelsmeindivid,labels17individ))

print(np.column_stack(np.unique(all_individ_labels,return_counts=True)))
x = np.vstack(np.unique(all_individ_labels,return_counts=True))
np.savetxt('./Count_all_individ_Labels.csv',x,delimiter=",", fmt="%s")

all_individ_LLDs=np.vstack((mudiLLDs,meLLDs,me17LLDs))
del(labelsmuindivid,labelsmeindivid,labels17individ)
all_individ_LLDs = np.column_stack((all_individ_LLDs,all_individ_labels))
#check
all_individ_labels = pd.factorize(all_individ_LLDs[:,500])[0]
all_individ_LLDs = np.delete(all_individ_LLDs,[500],axis=1)
all_individ_LLDs = np.column_stack((all_individ_LLDs,all_individ_labels))
np.savetxt('./all_individ_LLDs.csv',all_individ_LLDs,delimiter=",", fmt="%s")
'''
labelsmubreed = np.ones([6614,1])
labelsmebreed = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,7]).astype(str)
labelsmebreed = np.reshape(labelsmebreed,(len(labelsmebreed),1))
labels17breed = (pd.read_csv('./mescalina2017.csv', header=None).values[:,4]).astype(str)
labels17breed = np.reshape(labels17breed,(len(labels17breed),1))
all_breed_labels=np.vstack((labelsmubreed,labelsmebreed,labels17breed))
print(np.column_stack(np.unique(all_breed_labels,return_counts=True)))
all_breed_LLDs=np.vstack((mudiLLDs,meLLDs,me17LLDs))
del(labelsmubreed,labelsmebreed,labels17breed)
all_breed_LLDs = np.column_stack((all_breed_LLDs,all_breed_labels))
print(all_breed_LLDs.shape)
#check
all_breed_LLDs = all_breed_LLDs[~(all_breed_LLDs[:,500]=='?'),:]
print(all_breed_LLDs.shape)
print(pd.factorize(all_breed_LLDs[:,500])[1])
all_breed_labels = pd.factorize(all_breed_LLDs[:,500])[0]
all_breed_LLDs = np.delete(all_breed_LLDs,[500],axis=1)
all_breed_LLDs = np.column_stack((all_breed_LLDs,all_breed_labels))
np.savetxt('./all_breed_LLDs.csv',all_breed_LLDs,delimiter=",", fmt="%s")

labelsmesex = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,5]).astype(str)
labelsmesex = np.reshape(labelsmesex,(len(labelsmesex),1))
labels17sex = (pd.read_csv('./mescalina2017.csv', header=None).values[:,3]).astype(str)
labels17sex = np.reshape(labels17sex,(len(labels17sex),1))
all_sex_labels=np.vstack((labelsmesex,labels17sex))
print(np.column_stack(np.unique(all_sex_labels,return_counts=True)))
all_breed_LLDs=np.vstack((mudiLLDs,meLLDs,me17LLDs))
all_sex_LLDs=np.vstack((meLLDs,me17LLDs))
del(labelsmesex,labels17sex)
all_sex_LLDs = np.column_stack((all_sex_LLDs,all_sex_labels))
print(all_sex_LLDs.shape)
#check
all_sex_LLDs = all_sex_LLDs[~(all_sex_LLDs[:,500]=='?'),:]
print(all_sex_LLDs.shape)
print(pd.factorize(all_sex_LLDs[:,500])[1])
all_sex_labels = pd.factorize(all_sex_LLDs[:,500])[0]
all_sex_LLDs = np.delete(all_sex_LLDs,[500],axis=1)
all_sex_LLDs = np.column_stack((all_sex_LLDs,all_sex_labels))
np.savetxt('./all_sex_LLDs.csv',all_sex_LLDs,delimiter=",", fmt="%s")

labelsmuage = (pd.read_csv('./mudi_labels.csv', header=None).values[:,5]).astype(str)
labelsmuage = np.reshape(labelsmuage,(len(labelsmuage),1))
labelsmeage = (pd.read_csv('./mescalina_labels.csv', header=None).values[:,6]).astype(str)
labelsmeage = np.reshape(labelsmeage,(len(labelsmeage),1))
all_age_labels=np.vstack((labelsmuage,labelsmeage))
print(np.column_stack(np.unique(all_age_labels,return_counts=True)))
all_age_LLDs=np.vstack((mudiLLDs,meLLDs))
del(labelsmuage,labelsmeage)
all_age_LLDs = np.column_stack((all_age_LLDs,all_age_labels))
print(all_age_LLDs.shape)
#check
all_age_LLDs = all_age_LLDs[~(all_age_LLDs[:,500]=='?'),:]
print(all_age_LLDs.shape)
print(pd.factorize(all_age_LLDs[:,500])[1])
all_age_labels = pd.factorize(all_age_LLDs[:,500])[0]
all_age_LLDs = np.delete(all_age_LLDs,[500],axis=1)
all_age_LLDs = np.column_stack((all_age_LLDs,all_age_labels))
np.savetxt('./all_age_LLDs.csv',all_age_LLDs,delimiter=",", fmt="%s")

del(mudiLLDs,meLLDs,me17LLDs)
'''