from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64
from kneed import DataGenerator, KneeLocator


# Use the Agg backend for Matplotlib to handle plotting without a GUI
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Configure the upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check for uploaded file
        uploaded_file = request.files.get('file_upload')
        
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_location)
            flash(f"File uploaded successfully to {file_location}", 'success')
            
            try:
                # Process the uploaded file
                analysis_results, plots, gender_pie_charts = process_file(file_location)
                return render_template('results.html', analysis_results=analysis_results, plots=plots, gender_pie_charts=gender_pie_charts)
            except Exception as e:
                flash(f"Error processing file: {e}", 'error')
                print(f"Error processing file: {e}")
        else:
            flash("Invalid file type. Please upload a CSV file.", 'error')

    return render_template('index.html')

def process_file(file_path):
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError("Error reading the CSV file: " + str(e))

    # Check if required columns are present
    required_columns = ['id', 'Age', 'Flight Distance', 'Seat comfort', 'Cleanliness']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("CSV file is missing required columns: " + ", ".join(required_columns))

    # Select the desired features for clustering
    X = df[['Age', 'Flight Distance', 'Seat comfort', 'Cleanliness']]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate WCSS
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    # Optimal cluster
    optimal_k = determine_optimal_clusters(wcss)
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    df['cluster'] = y_kmeans

    # Analyze cluster summary
    segment_summary = df.groupby('cluster').agg({
    'Age': 'mean',
    'Flight Distance': 'mean',
    'Seat comfort': 'mean',
    'Cleanliness': 'mean',
    'id': 'count'
    }).rename(columns={
    'Age': 'Rata-rata Usia',
    'Flight Distance': 'Rata-rata Jarak Penerbangan',
    'Seat comfort': 'Rata-rata Kenyamanan Kursi',
    'Cleanliness': 'Rata-rata Kebersihan',
    'id': 'Jumlah Data'
    })

    plots = []

    # Histogram plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for i, col in enumerate(X.columns):
        sns.histplot(X[col], ax=axs[i])
        axs[i].set_title('Histogram of ' + col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Elbow plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
    plt.xlabel("K Value")
    plt.xticks(range(1, 11))
    plt.ylabel("WCSS")
    plt.title("Elbow Method For Optimal k")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Scatter plot Age vs Flight Distance dengan centroid
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df['Age'],
        y=df['Flight Distance'],
        hue=df['cluster'],
        palette=sns.color_palette("tab10", n_colors=3),  # pastikan jumlah sesuai cluster
        s=60
    )

    # Tambahkan centroid
    centroids = kmeans.cluster_centers_
    # Ambil indeks kolom Age dan Flight Distance dari X.columns
    age_idx = list(X.columns).index('Age')
    flight_idx = list(X.columns).index('Flight Distance')

    # Karena centroid diambil dari data yang sudah diskalakan (standardized), kita perlu inverse transform:
    centroids_original = scaler.inverse_transform(centroids)

    # Plot titik centroid
    plt.scatter(
        centroids_original[:, age_idx], centroids_original[:, flight_idx],
        s=200, c='black', marker='X', label='Centroid'
    )

    plt.xlabel('Usia')
    plt.ylabel('Jarak Penerbangan')
    plt.title('Klaster: Usia vs Jarak Penerbangan')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # 3D Plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(optimal_k):
        ax.scatter(df['Age'][df.cluster == i],
                   df['Flight Distance'][df.cluster == i],
                   df['Seat comfort'][df.cluster == i],
                   s=60, label=f'Cluster {i}')
    ax.view_init(35, 185)
    plt.xlabel("Age")
    plt.ylabel("Flight Distance")
    ax.set_zlabel('Seat comfort')
    plt.title('3D Clustering Plot')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Tidak ada gender pie chart karena kolom Gender sudah tidak dipakai
    gender_pie_charts = []

    return segment_summary, plots, gender_pie_charts

def determine_optimal_clusters(wcss):
    """ Determine the optimal number of clusters using the Elbow method. """
    # You can adjust the logic here if needed
    kneedle = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    return kneedle.elbow

if __name__ == '__main__':
    app.run(debug=True)

