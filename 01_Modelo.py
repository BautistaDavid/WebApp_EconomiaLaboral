import streamlit as st
import pickle as pkl

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
    edad = st.number_input('Edad:',min_value=(13))
    años_edc = st.number_input('Años de Educación:',min_value=(0))
    pareja = st.selectbox('¿Tiene Pareja?',('Si','No'))
        
with col2:
    estudiante_act = st.selectbox('¿Actualmente está estudiando?',('Si','No'))
    estrato = st.number_input('Seleccione el Estrado de su Vivienda:',min_value=0,max_value=6)
    pc = st.selectbox('¿Tiene una computadora?',('Si','No'))
    internet = st.selectbox('Tiene acceso a internet en su vivienda?',('Si','No'))

# Open clf model  
clf_pickle = open('model/clf_randomforest.pickle','rb')
clf = pkl.load(clf_pickle)
clf_pickle.close()

# Opeen Scaler model 
scaler_pickle = open('model/clf_randomforest.pickle','rb')
scaler = pkl.load(scaler_pickle) 
scaler_pickle.close()

    














# st.markdown(
# """
# <p><span style="font-size: 18px;">Esta aplicación web presenta un formulario para que ingreses con tu propia información o la que desees,\
#  esto con el fin de estimar la probabilidad de que estes empleado en la Ciudad de Bogotá, Colombia.</span></p>

# Esta estimación se realiza por medio del uso de un algoritmo de machine Learning llamado Random Forest\
#  (En español: Bosque Aleatorio 😜), estos son un método de clasificación derivado del uso simultaneo de los\
#  también famosos decision trees (Arboles de decision 🌲). En las 

# """
# )

st.sidebar.markdown("""# **¡Navega por esta App!** ☝️☝️""")

# page_names_to_funcs = {
#     "Main Page": main_page,
#     "Page 2": page2,
#     "Page 3": page3,
# }