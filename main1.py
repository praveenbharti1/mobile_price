import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")

# Placeholder function for prediction and interpretation
def predict_activity(battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi):
    # Load the pre-trained model and scaler from the saved file
    with open('randomforest.pkl', 'rb') as model_file:
        randomforest_model = pickle.load(model_file)
    
    
    st.write('hello')
    with open('scaled_data.pkl', 'rb') as model_file:
        scaler_model = pickle.load(model_file)
    # Extract the loaded model and scaler
    
    
    # Scale the input data using the loaded scaler
    scaled_data = scaler_model.transform([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])
   
    # Make a prediction using the loaded random forest model
    prediction = randomforest_model.predict(scaled_data)
    
    return prediction

# Prediction button
def interpret_prediction(prediction):
    if prediction == 0:
        return "Low Cost"
    elif prediction == 1:
        return "Medium Cost"
    elif prediction == 2:
        return "High Cost"
    elif prediction == 3:
        return "Very High Cost"
    else:
        return "Unknown"
    


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable Matplotlib warning

    # Header for the web app
    html_temp = """
    <div style="background-color:#2E86C1; padding:15px; text-align:center;">
    <h2 style="color:white; font-family: 'Arial', sans-serif;">Phone Price Prediction App</h2>
    </div>

    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar inputs for model parameters
    st.header("Input Parameters")
    battery_power = st.number_input("Enter Battery Power", min_value=0.0, max_value=6000.0, value=0.0)
    blue = st.selectbox("Bluetooth", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    blue = 1 if blue == "Yes" else 0

    clock_speed = st.number_input("Enter Clock Speed", min_value=0.0, max_value=4.0, value=0.0)
    dual_sim = st.selectbox("Dual sim", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    dual_sim = 1 if dual_sim == "Yes" else 0

    fc = st.number_input("Number of Front Cameras", min_value=0, max_value=5, value=0)
    four_g = st.selectbox("Four_g", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    four_g = 1 if four_g == "Yes" else 0

    int_memory = st.number_input("Internal Memory", min_value=0, max_value=512, value=0)
    m_dep = st.number_input("Mobile Depth (cm)", min_value=0.0, max_value=1.0,format="%f", value=0.0)
    mobile_wt = st.number_input("Mobile Weight", min_value=0.0, max_value=250.0, value=0.0)
    n_cores = st.number_input("Number of Cores", min_value=0, max_value=8, value=0)
    pc = st.number_input("Primary Camera", min_value=0, max_value=3, value=0)
    px_height = st.number_input("Pixel Resolution Height", min_value=0, max_value=2000, value=0)
    px_width = st.number_input("Pixel Resolution Width", min_value=0, max_value=2000, value=0)
    ram = st.number_input("RAM", min_value=0, max_value=12, value=0)
    sc_h = st.number_input("Screen Height", min_value=0.0, max_value=20.0, value=0.0)
    sc_w = st.number_input("Screen Width", min_value=0.0, max_value=20.0, value=0.0)
    talk_time = st.number_input("Talk Time", min_value=0, max_value=30, value=0)
    three_g = st.selectbox("3G Support", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    three_g = 1 if three_g == "Yes" else 0
    
    touch_screen = st.selectbox("Touch Screen", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    touch_screen = 1 if touch_screen == "Yes" else 0

    wifi = st.selectbox("Wi-Fi", ["No", "Yes"], index=1)  # Default index set to 1 for "Yes"
    wifi = 1 if wifi == "Yes" else 0

    # Prediction button
    if st.button("Predict"):
        prediction = predict_activity(battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi)
        result = interpret_prediction(prediction[0])
        st.success(f"The Person is: {result}")

    # Project Overview section
    st.subheader("Embark on a Mobile Companion")

    # Project introduction
    st.write("Explore Phone Price Prediction: Where Tech Meets Analytics!")
    st.write("📱  Unlock mobile pricing insights with machine learning, decoding intricate patterns in device features for accurate predictions.")
    st.write("💡 Embark on a data-driven journey into the mobile device realm, where each feature contributes to the symphony of pricing. Happy predicting! 🌟")

    # Dynamic line plot showing variation of activity against features
    st.subheader("Decoding the Symphony of Features 🎵📱")

    # features = ["Battery Power", "Clock Speed", "Dual SIM", "Front Cameras", "4G Support", "Internal Memory", "Mobile Depth", "Mobile Weight", "Number of Cores", "Primary Camera", "Pixel Resolution Height", "Pixel Resolution Width", "RAM", "Screen Height", "Screen Width", "Talk Time"]

    # # Simulate a dynamic dance of data
    # activity_values = np.random.rand(len(features))

    # Create a vibrant visual representation
    # plt.plot(features, activity_values, marker='o', color='#3498db')  # Blue color
    # plt.xlabel("Mobile Features")
    # plt.ylabel("Activity Level")
    # plt.title("Harmony of Features: Unleashing Mobile Insights 📊")
    # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    # st.pyplot(plt)

    # Contact section
    st.subheader("Contact 📧")
    st.write("Contact us at us at [rajputabhishek4ever9@gmail.com]")

if __name__ == '__main__':
    main()