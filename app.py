# Import statements:
from PIL import Image
from flask import Flask, render_template, request
import numpy as np
from sklearn.cluster import KMeans

# Initialize the Flask application:
app = Flask(__name__)


# Create a route for the home("index") page:
@app.route('/')
def index():
    return render_template('index.html')


# Create a route for the "results.html" page:
@app.route('/', methods=['POST'])
def upload():
    file = request.files['image']
    image = Image.open(file)

    # Resize the uploaded image to speed up clustering:
    image = image.resize((500, 500))

    # Convert the uploaded image to a numpy array:
    im_arr = np.array(image)

    # Reshape array to 2D (pixel rows as samples, RGB values as features):
    im_arr = im_arr.reshape(-1, 3)

    # Apply k-means clustering with k=16:
    kmeans = KMeans(n_clusters=16, random_state=0).fit(im_arr)

    # Get the 16 most representative colors in the uploaded image:
    colors = kmeans.cluster_centers_.astype(int)

    # Convert RGB values to hex codes:
    hex_codes = ['#' + ''.join(f'{c:02X}' for c in color) for color in colors]

    # Create swatches for the results table:
    swatches = [f'#{c[1][0]:02X}{c[1][1]:02X}{c[1][2]:02X}' for c in
                enumerate(colors)]

    return render_template('result.html', hex_codes=hex_codes,
                           swatches=swatches)


if __name__ == '__main__':
    app.run(debug=True)
