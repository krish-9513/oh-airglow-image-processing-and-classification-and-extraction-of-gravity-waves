# OH-Airglow-Image-Processing-and-Classification-and-extraction-of-Gravity-Waves
AI-based airglow image analysis system using preprocessing, CNN classification, and visualization. Processes FITS images with noise reduction techniques and classifies them into moon, cloud, glare, and clear sky categories using Python, TensorFlow, OpenCV, and Streamlit.
.Venv(Directory): includes all the dependencies
Model(Directory): stored trained CNN Model.
Preprocessing(Directory): Contains the files used for Preprocessing.

1. csv_loader.py – loads Csv files.
2. cosmic.py – logic of how to remove cosmic ray using median filter.
3. contrast.py – code of percentile stretch for visualization.
4. normalize.py – normalization values

convert_csv_to_npy.py – convert all the csv files to numpy array including
darkframe file and flatfield file.
darkframe.npy – darkframe in npy format.
flatfieldframe.npy - flatfield in npy format.
run_pipeline_save_processed_npy.py – loop through whole year folder, load csv,
convert into npy , apply preprocessed steps like removal of cosmic rays, dark frame
subtraction, flatfield correction, and save it to output folder path.
Classification.py - Take npy files from numpy folders. Use model structure and
classify images using trained 2D CNN model. Output will be stored as
year/month/nights/4-txt files(clear,cloudy,glare and moon) which include name of npy
files.
Differencing.py – matches time from name of npy files ( from classified folder
structure) and take time diff of 60 sec, if not present then immediate previous.
apply high pass filtering for smooth background and wave contrast to make wave
visible.
Differencing_video.py – combines frame and make video to visualize waves.
Web(directory): directly takes images from folder and display.

1. Images(directory) – includes images to show on website.
2. Img(directory)- includes graph to display.
3. Pages(directory) – includes code for 4 diff pages(
SIRI,NIRIS,PAIRS,CMAP).
4. Home.py – main page for website ( to run web use streamlit run
Home.py).

Web_app(directory): display graph using classified folder.

1. Images(directory) – includes images to show on website.
2. Pages(directory) – includes code for 4 diff pages(
SIRI,NIRIS,PAIRS,CMAP)
3. Home.py – main page for website ( to run web use streamlit run
Home.py).
