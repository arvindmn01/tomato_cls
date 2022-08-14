import streamlit as st
import cv2
import numpy as np 
import tensorflow as tf

model=tf.keras.models.load_model('tomato/1')

st.title('**WELCOME**')
st.header('you can classify disease in tomato leaf')
file=st.file_uploader('choose a file')

exe_=st.button('predict')

def prediction_(img,model):
    image_arr=cv2.cvtColor(
        cv2.imdecode(np.fromstring(img,dtype='uint8'),1),
        cv2.COLOR_BGR2RGB,
    )
    image_arr=np.expand_dims(image_arr,axis=0)

    predictions=model.predict(image_arr)
    class_names=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
    predict_class=class_names[np.argmax(predictions[0])]
    confidence=round(100 * (np.max(predictions[0])),2)
    return image_arr,predict_class,confidence
image=''

if exe_:
    image=file.getvalue()
    image_arr,predict_class,confidence=prediction_(image,model)
    st.write('PREDICTED: ',predict_class)
    st.write('CONFIDENCE: ',confidence)
    st.image(image_arr,caption=' ORIGINAL IMAGE')

