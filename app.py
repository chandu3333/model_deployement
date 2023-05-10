import streamlit as st
import numpy as np
from pickle import load
import pickle
import pandas as pd
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resources")
model_path= os.path.join(dir_of_interest,"model.pkl")
ohe_path= os.path.join(dir_of_interest,"ohe.pkl")

st.header("welcome let's predict our mobile price ranges :")

ohen = pickle.load(open(ohe_path,'rb'))
knm = pickle.load(open(model_path,"rb"))


battery_power = st.text_input("enter battery range",placeholder="enter value")
blue = st.selectbox('enter blue Value', ('NO', 'yes'))
#st.text_input("enter any blue",placeholder="enter value")
clock_speed = st.text_input("enter clock speed",placeholder="enter value")
dual_sim = st.selectbox('dualsim', ('NO', 'yes'))
#st.text_input("enter dual sim value",placeholder="enter value")
fc = st.text_input("enter any fc value",placeholder="enter value")
four_g  = st.selectbox('enter four_g', ('NO', 'yes'))
#st.text_input("enter four_g",placeholder="enter value")
int_memory = st.text_input("enter int_memory",placeholder="enter value")
m_dep = st.text_input("enter m_dep",placeholder="enter value")
mobile_wt = st.text_input("enter any mobile_wt",placeholder="enter value")
n_cores  = st.selectbox('enter n_cores value', (4, 7, 8, 3, 2, 5, 1, 6))
#st.text_input("enter n_cores",placeholder="enter value")
pc = st.text_input("enter pc",placeholder="enter value")
px_height = st.text_input("enter px_height",placeholder="enter value")
px_width = st.text_input("enter px_width",placeholder="enter value")
ram = st.text_input("enter ram",placeholder="enter value")
sc_h = st.text_input("enter sc_h",placeholder="enter value")
sc_w = st.text_input("enter any sc_w",placeholder="enter value")
talk_time  = st.text_input("enter talk_time",placeholder="enter value")
three_g = st.selectbox('enter three_g valuue', (0, 1))
#st.text_input("enter three_g",placeholder="enter value")
touch_screen  = st.selectbox('enter touch screen', (0, 1))
#st.text_input("enter touch_screen",placeholder="enter value")
wifi = st.selectbox('enter wifi', (0, 1))
#st.text_input("enter wifi",placeholder="enter value")

btn = st.button("Predict")
col_names = ['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi','blue_NO','blue_yes','dual_sim_NO','dual_sim_yes','four_g_NO','four_g_yes']
if btn == True:
    if battery_power and blue and clock_speed and dual_sim and fc and four_g and int_memory and clock_speed and m_dep and mobile_wt and n_cores and pc and px_height and px_width and sc_h and sc_w and talk_time and three_g and touch_screen and wifi:
        q_p = pd.DataFrame([[int(battery_power), float(clock_speed), int(fc), int(int_memory), float(m_dep), float(mobile_wt), int(n_cores), int(pc), float(px_height), float(px_width), float(ram), int(sc_h), int(sc_w), int(talk_time), int(three_g),int(touch_screen), int(wifi)]])
        q_p1 = pd.DataFrame([[blue,dual_sim,four_g]])
        q_p2 = ohen.transform(q_p1).toarray()
        q_p3 = pd.concat([q_p,pd.DataFrame(q_p2)],axis=1)
        q_p3.columns=col_names
        pred = knm.predict(q_p3)
        st.subheader("predicted mobile range in  :")
        st.success(pred)
else:
        st.error("enter the value properly.")

#joblib.dump(kmodel,open("final.joblib","wb"))
