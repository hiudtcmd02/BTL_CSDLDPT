import click
import os
import utils
import io
import pickle
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask.cli import with_appcontext
from skimage import transform, color
from skimage.feature import hog
from sklearn.metrics import mean_squared_error
from PIL import Image
from scipy.spatial import KDTree

app = Flask(__name__, static_url_path="", static_folder="public")
cors = CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///video_searching.db"
app.config["CORS_HEADERS"] = "Content-Type"

db = SQLAlchemy(app)


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    path = db.Column(db.String, nullable=False)
    frames = db.relationship("Frame", backref="video")


class Frame(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    feature_vector = db.Column(db.PickleType, nullable=False)
    time = db.Column(db.BigInteger, nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey("video.id"), nullable=False)


@click.command("create-table")
@with_appcontext
def create():
    db.create_all()


@click.command("extract-featured-video")
@with_appcontext
def extract_featured_video():
    videos_path = "public/videos"
    count = 1
    for file in os.listdir(videos_path):
        print(f"{count}: {file} starting")
        count += 1

        file_path = os.path.join(videos_path, file)
        video = Video.query.filter_by(path=f"videos/{file}").first()
        if video:
            print(f"{file} done!\n")
            continue
        video = Video(path=f"videos/{file}")
        db.session.add(video)

        print("cutting...")
        frames = utils.cut_video_into_frames(file_path)

        print("resizing to 512x512px...")
        frames = [transform.resize(frame, (512, 512)) for frame in frames]

        print("converting to gray image...")
        frames = [color.rgb2gray(frame) for frame in frames]

        print("extracting feature vector...")
        length = len(frames)
        for i in range(0, len(frames), 6):
            frames_of_second = frames[i : i + 6 if i + 6 <= length else length]
            feature_vectors = []
            for frame in frames_of_second:
                fd, hog_img = hog(
                    frame,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                )
                feature_vectors.append(fd)
            frame = Frame(
                feature_vector=utils.closest_to_mean(feature_vectors),
                time=i // 6,
                video=video,
            )
            db.session.add(frame)
        db.session.commit()
        print(f"{file} done!\n")


@click.command("save-kd-tree")
@with_appcontext
def save_kd_tree():
    frames = Frame.query.all()
    kdtree = [frame.feature_vector for frame in frames]
    kdtree = KDTree(kdtree)
    with open("instance/kd-tree.pk", "wb") as f:
        pickle.dump(kdtree, f)


app.cli.add_command(create)
app.cli.add_command(extract_featured_video)
app.cli.add_command(save_kd_tree)


@app.route("/search", methods=["POST"])
@cross_origin()
def search():
    if "file" not in request.files:
        return jsonify({"error": True, "message": "file not provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": True, "message": "no file selected"}), 400
    try:
        image_bytes = file.read()
        image_file = io.BytesIO(image_bytes)
        image = np.array(Image.open(image_file))
        image = transform.resize(image, (512, 512, 3))
        image = color.rgb2gray(image)
        feature_vector, hog_img = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )
        file = open("instance/kd-tree.pk", "rb")
        kdtree = pickle.load(file)
        file.close()
        frames = Frame.query.all()
        distance, index = kdtree.query(feature_vector)
        nearest_frame = frames[index]

        return jsonify(
            {
                "error": False,
                "data": {
                    "path": f"{request.host_url}{nearest_frame.video.path}",
                    "time": nearest_frame.time,
                },
            }
        )
    except Exception as ex:
        return jsonify({"error": True, "message": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=3003, debug=True)
