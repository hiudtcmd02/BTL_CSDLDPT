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

# # Đọc vào file kdtree2 những frame không thuộc video đã tìm thấy
# def save_another_kd_tree(x):
#     frames = Frame.query.all()
#     kdtree2 = []
#     for frame in frames:
#         if frame.video_id not in x:
#             kdtree2.append(frame.feature_vector)
#     kdtree2 = KDTree(kdtree2)
#     with open("instance/kd-tree2.pk", "wb") as f:
#         pickle.dump(kdtree2, f)

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
        quantity_fw = len(feature_vector)
        distance, index = kdtree.query(feature_vector, k = quantity_fw)
        a = []
        nearest_frame = []
        count = 0
        for i in index:
            if frames[i].video_id not in a:
                nearest_frame.append(frames[i])
                a.append(frames[i].video_id)
                count += 1
            if(count == 3):
                break
        # print(f"video1: {a[0]}")
        # print(f"video2: {a[1]}")
        # print(f"video3: {a[2]}")
        
        # second_nearest_frame = frames[index[1]]
        # third_nearest_frame = frames[index[2]]

        # # Tạo mảng chưa video_id đã lấy
        # x = []
        # x.append(nearest_frame.video_id)

        # #Lấy ra frame của video thứ 2
        # save_another_kd_tree(x)
        # file = open("instance/kd-tree.pk", "rb")
        # kdtree2 = pickle.load(file)
        # file.close()
        # distance2, index2 = kdtree2.query(feature_vector)
        # second_nearest_frame = frames[index2]
        # x.append(second_nearest_frame.video_id)
        # print(f"index2: {index2}")

        # #Lấy ra frame của video thứ 3
        # save_another_kd_tree(x)
        # file = open("instance/kd-tree.pk", "rb")
        # kdtree3 = pickle.load(file)
        # file.close()
        # distance3, index3 = kdtree3.query(feature_vector)
        # third_nearest_frame = frames[index3]
        # x.append(third_nearest_frame.video_id)
        # print(f"index3: {index3}")

        return jsonify(
            {
                "error": False,
                "data": {
                    "path": f"{request.host_url}{nearest_frame[0].video.path}",
                    "time": nearest_frame[0].time,

                    # truyền thêm dữ liệu của 2 video gần nhất
                    "second_path": f"{request.host_url}{nearest_frame[1].video.path}",
                    "second_time": nearest_frame[1].time,
                    "third_path": f"{request.host_url}{nearest_frame[2].video.path}",
                    "third_time": nearest_frame[2].time,
                },
            }
        )
    except Exception as ex:
        return jsonify({"error": True, "message": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=3003, debug=True)
