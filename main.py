## Streamlit file to show predictions.
import streamlit as st
import pickle
import datetime

# Load the model
with open("C:\\Users\\DELL\\PycharmProjects\\singaporeresale\\resale_model.pkl", 'rb') as f:
  model = pickle.load(f)

# Categorical variable mappings
categorical_mappings = {
    'town': {'SENGKANG': 21, 'PUNGGOL': 18, 'WOODLANDS': 25, 'YISHUN': 26,
             'TAMPINES': 23, 'JURONG WEST': 13, 'BEDOK': 1, 'HOUGANG': 11,
             'CHOA CHU KANG': 8, 'ANG MO KIO': 0, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
             'BUKIT BATOK': 3, 'TOA PAYOH': 24, 'PASIR RIS': 17, 'KALLANG/WHAMPOA': 14,
             'QUEENSTOWN': 19, 'SEMBAWANG': 20, 'GEYLANG': 10, 'CLEMENTI': 9,
             'JURONG EAST': 12, 'BISHAN': 2, 'SERANGOON': 22, 'CENTRAL AREA': 7,
             'MARINE PARADE': 16, 'BUKIT TIMAH': 6, 'LIM CHU KANG': 15},

    'flat_type': {'4 ROOM': 3, '5 ROOM': 4, '3 ROOM': 2,
                  'EXECUTIVE': 5, '2 ROOM': 1, 'MULTI-GENERATION': 6,
                  '1 ROOM': 0},

    'storey_range': {'04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '01 TO 03': 0,
                     '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                     '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                     '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                     '49 TO 51': 16},

    'flat_model': {'Model A': 8, 'Improved': 5, 'New Generation': 12, 'Premium Apartment': 13,
                   'Simplified': 16, 'Apartment': 3, 'Maisonette': 7, 'Standard': 17,
                   'DBSS': 4, 'Model A2': 10, 'Model A-Maisonette': 9, 'Adjoined flat': 2,
                   'Type S1': 19, 'Type S2': 20, 'Premium Apartment Loft': 14, 'Terrace': 18,
                   'Multi Generation': 11, '2-room': 0, 'Improved-Maisonette': 6, '3Gen': 1,
                   'Premium Maisonette': 15},
}


def convert():
    time = datetime.datetime.now()
    x = time.year
    y = str(x)[0]
    z = int(y)
    return x


# Input widgets for user interaction
st.title("House Price Prediction App")

a = st.sidebar.selectbox('Enter town', options=categorical_mappings['town'])
b = st.sidebar.selectbox('Enter type of flat', options=categorical_mappings['flat_type'])
c = st.sidebar.selectbox('Enter Storey Range', options=categorical_mappings['storey_range'])
d = st.sidebar.number_input('Enter Floor Area(sqm)', min_value=0, max_value=1000)
e = st.sidebar.selectbox('Enter Flat Model', options=categorical_mappings['flat_model'])
f = st.sidebar.number_input('Enter Year of resale', min_value=0, max_value=3000)
g = st.sidebar.number_input('Enter Month of resale', min_value=1, max_value=12)
h = st.sidebar.number_input('Enter Lease Start Year', min_value=1900, max_value=3000,
                            help="Enter year which is less than 99 years from now since older flats are depracted "
                                 "by singapore govt as per their environmental policies"
                            )
if h:
    i = st.sidebar.number_input('Enter remaining lease Years', min_value=h-convert()+99, max_value=2130,
                                help='Singapore has 99 years of limit for any building so flats older than'
                                     ' 99 years are depracated and cant be resold'
                                )
j = st.sidebar.number_input('Enter remaining lease months', min_value=1, max_value=12)

input_data = [categorical_mappings['town'][a],
              categorical_mappings['flat_type'][b],
              categorical_mappings['storey_range'][c],
              d,
              categorical_mappings['flat_model'][e],
              f, g, h, i, j]

st.write(input_data)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)

    # Display the prediction result
    prediction_scale = np.exp(prediction[0])
    st.subheader("Prediction Result:")
    st.write(f"The predicted house price is: {prediction_scale:,.2f} /-")
