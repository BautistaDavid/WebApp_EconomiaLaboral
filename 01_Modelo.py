import streamlit as st
import pickle as pkl
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils import results


from model.clf_randomforest import X_test

st.title('Estimando la Probabilidad de Estar Empleado en Bogotá. 💼🍻')

st.markdown("""---""")
texto1 = '<p><span style="font-size: 21px; font-family: Helvetica;">Esta aplicación web presenta un formulario para que ingreses con tu propia información o la que desees,\
esto con el fin de estimar la probabilidad de que estes empleado en la Ciudad de Bogotá, Colombia.</span></p>\
<p><span style="font-size: 21px; font-family: Helvetica;"> Esta estimación se realiza por medio del uso de un algoritmo de machine Learning llamado Random Forest\
 (En español: Bosque Aleatorio 😜), estos son un método de clasificación derivado del uso simultaneo de los\
 también famosos Decision Trees (Arboles de decision 🌲).</span></p>\
<p><span style="font-size: 21px; font-family: Helvetica;"> En otras secciones de esta App podrá explorar mas sobre los datos usados y la construcción del modelo usando Python,\
 por el momento empecemos con lo interesante, ingrese la información que solicita el siguiente formulario y siga explorando la App.</span></p>'   

st.markdown(texto1,unsafe_allow_html=True)

col1,col2=st.columns(2)
with col1:
    sexo = st.selectbox('Genero:',('Hombre','Mujer'))
    edad = st.number_input('Edad:',min_value=(15))
    años_edc = st.number_input('Años de Educación:',min_value=(0))
    pareja = st.selectbox('¿Tiene Pareja?',('Si','No'))
    padres = st.selectbox('¿Con cual de sus padres reside?',('Padre','Madre','Ambos','Ninguno'))
        
with col2:
    estudiante_act = st.selectbox('¿Actualmente está estudiando?',('Si','No'))
    estrato = st.number_input('Seleccione el Estrado de su Vivienda:',min_value=0,max_value=6)
    pc = st.selectbox('¿Tiene una computadora?',('Si','No'))
    internet = st.selectbox('Tiene acceso a internet en su vivienda?',('Si','No'))
    etnia = st.selectbox('Pertenece usted a alguno de los siguientes grupos etnicos...Afrodecendiente, gitano, indigena, palenquero o\
        raizal',('Si','No'))
# configurin variables 

sexo = 1 if sexo == 'Hombre' else 0
pareja = 1 if pareja == 'Si' else 0
estudiante_act = 1 if estudiante_act == 'Si' else 0
pc = 1 if pc == 'Si' else 0
internet = 1 if internet == 'Si' else 0
sin_rec_etnico = 0 if etnia =='Si' else 1

if padres == 'Padre':
    reside_padre,  reside_madre = 1, 0
if padres == 'Madre':
    reside_padre, reside_madre = 0, 1
if padres == 'Ambos':
    reside_padre, reside_madre = 1,1
if padres == 'Ninguno':
    reside_padre, reside_madre = 0, 0


# Open clf model  
clf_pickle = open('model/clf_randomforest.pickle','rb')
clf = pkl.load(clf_pickle)
clf_pickle.close()

# Open Scaler model 
scaler_pickle = open('model/scaler.pickle','rb')
scaler = pkl.load(scaler_pickle) 
scaler_pickle.close()

X_test = pd.DataFrame([sexo,edad,años_edc,pareja,estudiante_act,estrato,pc,internet,reside_padre,reside_madre,sin_rec_etnico]).T
X_test[[1,2,5]] = scaler.transform(X_test[[1,2,5]])


pred = clf.predict_proba(X_test)
# st.write(pred[0,1])  Probabilidad de estar empleado

arr = [pred[0,0],pred[0,1]]
arr.sort()
fig, ax = plt.subplots(1,2,figsize=(14,6))
text_kwargs = dict(ha='center', va='center', fontsize=26, color='Black')
text_kwargs2 = dict(ha='center', va='center', fontsize=38, color='Black')
ax[0].text(0.6, 0.6, 'Se Estima Una Probabilidad\n De Estar Empleado En Bogotá De:',**text_kwargs)
ax[0].axis('off')
ax[1].text(0, 0, f'{round(pred[0,1]*100,2)}%',**text_kwargs2)
ax[1].axis('off')

v1 = [arr[0], arr[1],]
wedge_properties = {"width":0.3}
patches, _ = ax[1].pie(v1, wedgeprops=wedge_properties,colors=['gray','#5DADE2'])
patches[1].set_edgecolor('black')
patches[0].set_edgecolor('black')
plt.show()

st.pyplot(fig)

texto2 = '<p><span style="font-size: 21px; font-family: Helvetica;">Es pertinente que la muestra para la realización\
 del modelo se construyo a partir del manejo de la Gran Encuesta Integrada De Hogares (GEIH) presentada por el DANE para\
 el caso colombiano durante el año 2021.</span></p>\
 <p><span style="font-size: 21px; font-family: Helvetica;">Así mismo se debe comentar que las variable seleccionadas se apegan estrictamente\
 al análisis de características propias de un individuo, tales como edad, sexo, etnia, años de educación y las demás, por \
 lo cual el modelo logra estimar una probabilidad basada en los grupos de respuestas que aseguraron tener trabajo, sin embargo\
 no se está usando ningún tipo de herramienta o metodología para incluir dentro del modelo el analisis de dinámicas laborales\
 en Bogotá. </span></p>'

st.markdown(texto2,unsafe_allow_html=True)
st.markdown('## Metricas y parametros del Modelo')
texto3 = '<p><span style="font-size: 21px; font-family: Helvetica;">A continuación, se presenta la matriz de métricas y un heatmap de la matriz\
 de confusión del modelo dentro de la muestra de testeo. </span><p>'
st.markdown(texto3,unsafe_allow_html=True)




y_test = pd.read_csv('data/y_test.csv')
X_test = pd.read_csv('data/X_test.csv')

# y_test.drop(columns='Unnamed: 0',inplace=True)
y_pred = clf.predict(X_test)

st.markdown("""### Reporte de Metricas""")
st.table(pd.DataFrame(results(y_test,y_pred)))


fig = plt.figure(figsize=(4, 4))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d",xticklabels=['Desempleado','Empleado'],\
    yticklabels=['Desempleado','Empleado'],cbar='rainbow',);
plt.suptitle("Matriz De Confusión")
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()

st.pyplot(fig)















# st.markdown(
# """
# <p><span style="font-size: 18px;">Esta aplicación web presenta un formulario para que ingreses con tu propia información o la que desees,\
#  esto con el fin de estimar la probabilidad de que estes empleado en la Ciudad de Bogotá, Colombia.</span></p>

# Esta estimación se realiza por medio del uso de un algoritmo de machine Learning llamado Random Forest\
#  (En español: Bosque Aleatorio 😜), estos son un método de clasificación derivado del uso simultaneo de los\
#  también famosos decision trees (Arboles de decision 🌲). En las 

# """
# )

st.sidebar.markdown("""
# **¡Navega por esta App!** ☝️☝️""")

