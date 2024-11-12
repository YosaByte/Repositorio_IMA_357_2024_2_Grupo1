import streamlit as st
import pandas as pd
import numpy as np 
import math
import copy
from collections import Counter, OrderedDict
import nltk
from nltk import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

nltk.download('stopwords') 

stemmer = SnowballStemmer('spanish')
stop_words = nltk.corpus.stopwords.words('spanish') #stop words en español
stop_words_en = nltk.corpus.stopwords.words('english')
simbolos = (',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(',')', '[', ']','%','#', '$', '&','/')


tokenizer = TreebankWordTokenizer()

def BoW_vec(docs:list, tokenizer):
    doc_tokens = []
    for doc in docs:
        tokens = tokenizer.tokenize(doc.lower())
        tokens = [x for x in tokens if x not in stop_words and x not in stop_words_en]
        tokens = [stemmer.stem(x) for x in tokens]
        doc_tokens += [sorted(tokens)]
    all_doc_tokens = sum(doc_tokens, [])
    lexico = sorted(set(all_doc_tokens))
    zero_vector = OrderedDict((token, 0) for token in lexico)
    document_bow_vectors = []
    for i, doc in enumerate(docs):
        vec = copy.copy(zero_vector)
        tokens = tokenizer.tokenize(doc.lower())
        tokens = [x for x in tokens if x not in stop_words and x not in simbolos and x not in stop_words_en]
        tokens = [stemmer.stem(x) for x in tokens]
        token_counts = Counter(tokens)
        for key, value in token_counts.items():
            vec[key] = value
        document_bow_vectors.append(vec)
    return document_bow_vectors

def sim_coseno(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    dot_prod = 0
    frec=0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
        if v * vec2[i]!=0:
            frec+=1
    norm_1 = math.sqrt(sum([x**2 for x in vec1]))
    norm_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (norm_1 * norm_2), frec


#document_BoW_vector = BoW_vec(docs=documentos, tokenizer=tokenizer)


st.title('App item 3')

st.header('Tabla de noticias recopiladas de ElDiario')

#url='https://raw.githubusercontent.com/YosaByte/App/refs/heads/main/grupo_1.csv?token=GHSAT0AAAAAAC2GHJMHJ2HOA2D6TGX4IXU6ZZSTNRQ' 
df_news=pd.read_csv(r'grupo_1.csv',header=0) 

st.data_editor(df_news)

st.write('Escriba una palabra para determinar la noticia donde su palabra se reitere más veces y la frecuencia de esta misma.')

palabra=st.text_input('Escriba su palabra')

palabra_frec=df_news['Cuerpo'].tolist() 
palabra_frec.append(palabra)

word_BoW_vector=BoW_vec(docs=palabra_frec,tokenizer=tokenizer)

lista2=[]
for claves,valores in word_BoW_vector[30].items():
    lista2.append(valores)

id_palabra=lista2.index(max(lista2)) 
 
lista3=[]
for j in range(len(word_BoW_vector)):
    lista_aux=[]
    for i in word_BoW_vector[j].values():
        lista_aux.append(i)
    lista3.append(lista_aux)
frecuente=0
for lista_orden in lista3:
    if lista_orden[id_palabra]>frecuente:
        frecuente=lista_orden[id_palabra]
        id_p_cuerpo=lista3.index(lista_orden)
if id_p_cuerpo!=30:
    body={
        'Titular':df_news['Titular'][id_p_cuerpo],
        'Frecuencia':frecuente
        }
    result=pd.DataFrame(body,columns=['Titular','Frecuencia'],index=[id_p_cuerpo])
    st.write('Resultado de la plabra "'+palabra+'"')
    st.data_editor(result) 
else: 
    result='Pruebe otra palabra'
    #st.write('Resultado de la plabra "'+palabra+'"')
    st.write(result) 



st.write('Redacte una oración para encontrar la noticia donde su oración se reitere más veces')

texto=st.text_input('Escriba su oración')

documentos=df_news['Cuerpo'].tolist()

documentos.append(texto)

document_BoW_vector = BoW_vec(docs=documentos, tokenizer=tokenizer)

lista_cos=[]
for j,doc_1 in enumerate(document_BoW_vector):
        if j<30:
             lista_cos.append(sim_coseno(doc_1, document_BoW_vector[30]))
max_similitud=max(lista_cos)
id_maxima_sim=lista_cos.index(max_similitud)

lista2=[]
for claves,valores in document_BoW_vector[30].items():
    lista2.append(valores)

id_ora=[]
cont=0
for x in lista2:
    if x>0:
        id_ora.append(cont)
    cont+=1

lista3=[]
for j in range(len(document_BoW_vector)):
    lista_aux=[]
    lista_aux2=[]
    for i in document_BoW_vector[j].values():
        lista_aux.append(i)
    for k in id_ora:
        lista_aux2.append(lista_aux[k])
    lista3.append(sum(lista_aux2))

frecuente=0

for lista_orden in lista3:
    if lista_orden>frecuente:
        frecuente=lista_orden
        id_p_cuerpo=lista3.index(lista_orden)
if id_p_cuerpo!=30:
    
    st.write("La noticia que lleva por título "+df_news['Titular'][id_maxima_sim]+" tiene la mejor similutud coseno respecto a su oración")
    st.write("Similitud Coseno: "+max_similitud+".")
    
    body={
        'Titular':df_news['Titular'][id_p_cuerpo],
        'Frecuencia':frecuente
        }
    result=pd.DataFrame(body,columns=['Titular','Frecuencia'],index=[id_p_cuerpo])
    st.write('Resultado de la plabra "'+texto+'"')
    st.data_editor(result) 

else: 
    result='Pruebe otra oración'
    #st.write('Resultado de la plabra "'+palabra+'"')
    st.write(result) 
#result_word=pd.DataFrame(index=10,)


