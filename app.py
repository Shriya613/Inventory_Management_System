# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd
import sqlite3
import base64

# External packages
import streamlit as st

# Local Modules
import settings
import helper





conn = sqlite3.connect('inventory_database.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS products (key NUMERIC, time TEXT, name TEXT, countt NUMERIC)')

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("images/3.jpg")

img1 = get_img_as_base64("images/4.jpg")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:4/png;base64,{img1}");
background-size: cover;

}}

[data-testid="stHeader"] {{
background: rgba(0, 0, 0, 0);
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:3/png;base64,{img}");
background-position: center;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

#background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");



# Main page heading
#st.title("Object Detection using YOLOv8")
st.title("Inventory Management System")

date1=st.sidebar.date_input("Date")

# Sidebar
st.sidebar.header("ML Model Config")


# Model Options
model_type = st.sidebar.radio(
    "Select Categories", ["Noodles","Shampoo","Lays","Cooldrinks","Biscuits"])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0.0,1.0,0.5))






def get_pandas(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results[0].boxes.data.tolist()
  columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']

  # iterate through the list of boxes info and make some formatting
  for i in boxes_list:
    # round float xyxy coordinates:
    i[:4] = [round(i, 1) for i in i[:4]]
    # translate float class_id to an integer
    i[5] = int(i[5])
    # add a class name as a last element
    i.append(results[0].names[i[5]])

  # create the result dataframe
  columns.append('class_name')
  result_df = pd.DataFrame(boxes_list, columns=columns)

  return result_df




# Selecting Detection Or Segmentation
if model_type == 'Noodles':
    model_path = Path(settings.DETECTION_MODEL1)
elif model_type == 'Shampoo':
    model_path = Path(settings.DETECTION_MODEL2)
elif model_type == 'Lays':
    model_path = Path(settings.DETECTION_MODEL3)
elif model_type == 'Cooldrinks':
    model_path = Path(settings.DETECTION_MODEL4)
elif model_type == 'Biscuits':
    model_path = Path(settings.DETECTION_MODEL5)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Data/Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    st.markdown('<style>div[data-testid="stImageCaption"] { color: #080808; font-size: 20px; }</style>', unsafe_allow_html=True)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
                
                
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
               
                #st.write(boxes)
                
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        df=get_pandas(res)
                        count = df["class_name"].value_counts()
                        st.write(count)
                        if model_type == 'Noodles':
                            key=1
                        elif model_type == 'Shampoo':
                            key=2
                        elif model_type == 'Lays':
                            key=3
                        elif model_type == 'Cooldrinks':
                            key=4
                        elif model_type == 'Biscuits':
                            key=5
                

                        for i in range(len(df)):
                            c.execute('INSERT INTO products (key,time,name,countt) VALUES (?, ?, ?, ?) ',(key, date1, (df['class_name'].iloc[i]) , 1) )
                            conn.commit()
                                        
                        #for box in boxes:
                            #st.write(box.cls.tolist())
                except Exception as ex:
                    st.write(ex)
                    #st.write("No image is uploaded yet!")

                


elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

elif source_radio == settings.DATA:
    st.write("")
    st.subheader("Categories")
    choice=st.selectbox("Choose Mode",["All","Drinks","Lays","Shampoos","Biscuits","Noodles"])


    if choice=='All':

        read_list2=[]    
        c.execute('SELECT key,time,name,sum(countt) FROM products GROUP BY name ')
        read_list2=c.fetchall()
        df2 = pd.DataFrame(
            read_list2,
            columns=["Id","time","name","count"]
        )
        st.table(df2)

        
    elif(choice=='Lays'):
        

        read_list2=[]    
        c.execute('SELECT key,time,name,sum(countt) FROM products where key=3 GROUP BY name ')
        read_list2=c.fetchall()
        df2 = pd.DataFrame(
            read_list2,
            columns=["Id","time","name","count"]
        )
        st.table(df2)  


    elif(choice=='Shampoos'):


        read_list2=[]    
        c.execute('SELECT key,time,name,sum(countt) FROM products where key=2 GROUP BY name ')
        read_list2=c.fetchall()
        df2 = pd.DataFrame(
            read_list2,
            columns=["Id","time","name","count"]
        )
        st.table(df2)  
    
    elif(choice=='Drinks'):

        

        read_list2=[]    
        c.execute('SELECT key,time,name,sum(countt) FROM products where key=4 GROUP BY name ')
        read_list2=c.fetchall()
        df2 = pd.DataFrame(
            read_list2,
            columns=["Id","time","name","count"]
        )
        st.table(df2) 
            

    elif(choice=='Noodles'):

            read_list2=[]    
            c.execute('SELECT key,time,name,sum(countt) FROM products where key=1 GROUP BY name ')
            read_list2=c.fetchall()
            df2 = pd.DataFrame(
                read_list2,
                columns=["Id","time","name","count"]
            )
            st.table(df2)  

    else:
            read_list2=[]    
            c.execute('SELECT key,time,name,sum(countt) FROM products where key=5 GROUP BY name ')
            read_list2=c.fetchall()
            df2 = pd.DataFrame(
                read_list2,
                columns=["Id","time","name","count"]
            )
            st.table(df2)  


    choice5=st.selectbox("Download Mode",["None", "CSV","Excel"])

    if choice5=="CSV":
        
        st.download_button(label='download csv',data=df2.to_csv() ,mime='text/csv' ,)

    elif choice5=="Excel":
        
        writer = pd.ExcelWriter('products.xlsx')
        #df2.to_excel(writer, index=False)  # Save DataFrame to Excel file
        #writer.save()
        #st.download_button(label='download excel', data='products.xlsx', mime='text/xlsx')
        st.download_button(label='download excel',data=df2.to_excel(writer) ,mime='text/xlsx')
        #writer.save()

    #st.error("Please select a valid source type!")
