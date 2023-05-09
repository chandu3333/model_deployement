import streamlit as st
import numpy as np
from pickle import load
import pickle

ohen = pickle.load(open(r"C:\Users\HP\Desktop\ds_intership_2023\New folder\resources\ohe.pkl"))
knm = pickle.load(open(r"C:\Users\HP\Desktop\ds_intership_2023\New folder\resources\model.pkl"))


battery_power = st.text_input("enter battery range",placeholder="enter value")
blue = st.text_input("enter any blue",placeholder="enter value")
clock_speed = st.text_input("enter clock speed",placeholder="enter value")
dual_sim = st.text_input("enter dual sim value",placeholder="enter value")
fc = st.text_input("enter any fc value",placeholder="enter value")
four_g  = st.text_input("enter four_g",placeholder="enter value")
int_memory = st.text_input("enter int_memory",placeholder="enter value")
m_dep = st.text_input("enter m_dep",placeholder="enter value")
mobile_wt = st.text_input("enter any mobile_wt",placeholder="enter value")
n_cores  = st.text_input("enter n_cores",placeholder="enter value")
pc = st.text_input("enter pc",placeholder="enter value")
px_height = st.text_input("enter px_height",placeholder="enter value")
px_width = st.text_input("enter px_width",placeholder="enter value")
ram = st.text_input("enter ram",placeholder="enter value")
sc_h = st.text_input("enter sc_h",placeholder="enter value")
sc_w = st.text_input("enter any sc_w",placeholder="enter value")
talk_time  = st.text_input("enter talk_time",placeholder="enter value")
three_g = st.text_input("enter three_g",placeholder="enter value")
touch_screen  = st.text_input("enter touch_screen",placeholder="enter value")
wifi = st.text_input("enter wifi",placeholder="enter value")

btn = st.button("Predict")

if btn == True:
    if battery_power and blue and clock_speed and dual_sim and fc and four_g and int_memory and clock_speed and m_dep and mobile_wt and n_cores and pc and px_height and px_width and sc_h and sc_w and talk_time and three_g and touch_screen and wifi:
        q_p = np.array([int(battery_power), blue, float(clock_speed), dual_sim, int(fc), four_g, int(int_memory), float(m_dep), float(mobile_wt), int(n_cores), int(pc), float(px_height), float(px_width), float(ram), int(sc_h), int(sc_w), int(talk_time), int(three_g),int(touch_screen), int(wifi)])
        q_p2 = ohen.transform(q_p)
        pred = knm.predict(q_p2)
        st.success(pred)
    else:
        st.error("enter the value properly.")

#joblib.dump(kmodel,open("final.joblib","wb"))