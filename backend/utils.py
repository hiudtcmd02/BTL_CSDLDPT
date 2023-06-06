import cv2
import numpy as np
from sklearn.metrics import mean_squared_error


def cut_video_into_frames(video_path):
    # Đọc video
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        # Đọc từng frame
        success, frame = video_capture.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            frames.append(frame)
    # Giải phóng tài nguyên
    video_capture.release()

    return frames


def closest_to_mean(vectors):
    mean_vector = np.mean(vectors, axis=0)  # Tính trung bình của mảng vector

    distances = [
        mean_squared_error(vector, mean_vector) for vector in vectors
    ]  # Tính cosine similarity giữa mỗi vector và trung bình

    closest_index = np.argmin(
        distances
    )  # Lấy chỉ số của vector có khoảng cách nhỏ nhất

    closest_vector = vectors[closest_index]  # Lấy vector gần với trung bình nhất

    return closest_vector
